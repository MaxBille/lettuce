import torch
import numpy as np
import time

import lettuce as lt
from lettuce import LettuceException
from lettuce import torch_gradient
from lettuce.lattices import Lattice

__all__ = ["EquilibriumExtrapolationOutlet", "NonEquilibriumExtrapolationInletU", "SyntheticEddyInlet", "ZeroGradientOutlet"]

class EquilibriumExtrapolationOutlet(lt.AntiBounceBackOutlet):
    """Equilibrium outlet with extrapolated pressure and velocity from inside the domain
    """

    def __init__(self, lattice, direction):
        super(EquilibriumExtrapolationOutlet, self).__init__(lattice, direction)

    def __call__(self, f):
        here = [slice(None)] + self.index
        other = [slice(None)] + self.neighbor
        rho = self.lattice.rho(f)
        u = self.lattice.u(f)
        rho_w = rho[other]
        u_w = u[other]
        f[here] = self.lattice.equilibrium(rho_w[..., None], u_w[..., None])[..., 0]
        return f

    def make_no_stream_mask(self, f_shape):
        no_stream_mask = torch.zeros(size=f_shape, dtype=torch.bool, device=self.lattice.device)
        no_stream_mask[[np.setdiff1d(np.arange(self.lattice.Q), self.velocities)] + self.index] = 1
        return no_stream_mask

    def make_no_collision_mask(self, grid_shape):
        no_collision_mask = torch.zeros(size=grid_shape, dtype=torch.bool, device=self.lattice.device)
        no_collision_mask[self.index] = 1
        return no_collision_mask


class ZeroGradientOutlet(object):

    def __init__(self, lattice, direction):
        # assert (isinstance(direction, list) and len(direction) in [1,2,3] and ((np.abs(sum(direction)) == 1) and (np.max(np.abs(direction)) == 1) and (1 in direction) ^ (-1 in direction))), \
        #     LettuceException("Wrong direction. Expected list of length 1, 2 or 3 with all entrys 0 except one 1 or -1, "
        #                         f"but got {type(direction)} of size {len(direction)} and entrys {direction}.")
        self.direction = np.array(direction)
        self.lattice = lattice

        # select velocities to be replaced (the ones pointing against "direction")
        self.velocities = np.concatenate(np.argwhere(np.matmul(self.lattice.stencil.e, self.direction) < -1 + 1e-6),
                                         axis=0)  # alles was ENTGEGEN direction zeigt. Also z.B. als Auslass in positive X-Richtung alles, was eine Komponente mit -1 in x-Richtung hat.

        # build indices of u and f that determine the side of the domain
        self.index = []
        self.neighbor = []
        for i in self.direction:
            if i == 0:
                self.index.append(slice(None))
                self.neighbor.append(slice(None))
            if i == 1:
                self.index.append(-1)
                self.neighbor.append(-2)
            if i == -1:
                self.index.append(0)
                self.neighbor.append(1)

    def __call__(self, f):
        f[[self.velocities] + self.index] = f[[
                                                  self.velocities] + self.neighbor]  # alles, was "aus dem Auslass in Richtung Domäne zeigt" wird vom Nachbar übernommen; Dies kann einen Feedbackloop erzeugen...
        return f

    def make_no_stream_mask(self, f_shape):
        no_stream_mask = torch.zeros(size=f_shape, dtype=torch.bool, device=self.lattice.device)
        no_stream_mask[[self.velocities] + self.index] = 1
        return no_stream_mask


## STAND 29.05.24 läuft die NEX-Boundary MIT no_sreaming (alle pops. besser als > nur velocity_in),OHNE no_collision und OHNE Filter am stabilsten (Re2000 test, s.ohneNote MK Code 13./14./15.5.24)
class NonEquilibriumExtrapolationInletU(object):
    """ Guo's boundary condition
        https://www.researchgate.net/publication/230963379_Non-equilibrium_extrapolation_method_for_velocity_and_boundary_conditions_in_the_lattice_Boltzmann_method
        and LBM book page 189
        """

    def __init__(self, lattice, units, direction, u_w):
        # assert (isinstance(direction, list) and len(direction) in [1,2,3] and ((np.abs(sum(direction)) == 1) and (np.max(np.abs(direction)) == 1) and (1 in direction) ^ (-1 in direction))), \
        #     LettuceException("Wrong direction. Expected list of length 1, 2 or 3 with all entrys 0 except one 1 or -1, "
        #                         f"but got {type(direction)} of size {len(direction)} and entrys {direction}.")
        # print("start nonEQ_init")
        self.direction = np.array(direction)
        self.lattice = lattice
        self.u_w = units.convert_velocity_to_lu(self.lattice.convert_to_tensor(u_w))

        # select velocities to be bounced (the ones pointing in "direction")
        self.velocities_out = np.concatenate(np.argwhere(np.matmul(self.lattice.stencil.e, self.direction) > 1 - 1e-6),
                                             axis=0)  # alle, die einen Anteil in direction Richtung haben (nicht nur Betrag)
        # select velocities to be replaced (the ones pointing against "direction")
        self.velocities_in = np.concatenate(np.argwhere(np.matmul(self.lattice.stencil.e, self.direction) < -1 + 1e-6),
                                            axis=0)  # alle, die einen Anteil entgegen direction besitzen (nicht nur Betrag)

        # build indices of u and f that determine the side of the domain
        self.index = []
        self.neighbor = []
        for i in self.direction:
            if i == 0:
                self.index.append(slice(None))  # für diese Dimension "alles" ohne
                self.neighbor.append(slice(None))
            if i == 1:
                self.index.append(-1)  # für diese Dimension "letzter"
                self.neighbor.append(-2)  # für diese Dimension "vorletzter"
            if i == -1:
                self.index.append(0)  # für diese Dimension "erster"
                self.neighbor.append(1)  # für diese Dimension "zweiter"
        self.rho_old = 1.0
        if len(self.u_w.shape) > self.lattice.D:
            self.u_w = self.u_w[tuple([slice(None)] + self.index)]
        # print("u_w.shape in init NEQExtrapolInletU:", self.u_w.shape)

    def __call__(self, f):
        Tc = 100
        here = [slice(None)] + self.index  # q Platzhalter und Koordinaten der RB-Knoten
        other = [slice(None)] + self.neighbor  # q Platzhalter und Koordinaten der RB-Nachbarn
        # print("other in NEQEIU.call(): ", other)
        # print("here in NEQEIU.call(): ", here)
        # print("index in NEQEIU.call(): ", self.index)
        # print("f.shape:", f.shape, "f[others].shape:", f[other].shape)

        ## rho = self.lattice.convert_to_tensor(self.lattice.rho(f[other]))
        rho = self.lattice.convert_to_tensor(
            torch.sum(f[other], dim=0)[None, ...])  # berechne für alle Nachbarn die Dichte
        ## u = self.lattice.convert_to_tensor(self.lattice.u(f[other]))  # gibt nur die erste Ebene Knoten aus mit f[other]
        u = self.lattice.convert_to_tensor(
            torch.einsum("qd,q...->d...", self.lattice.e, f[other]) / rho)  # Geschwindigkeit auf den Nachbarknoten

        if self.u_w.shape == u.shape:
            # falls u_w schon die korrekte shape hat, nutze u_w
            u_w = self.u_w
        else:
            list = []
            for _ in u.shape: list += [1]  # pro Dimension in u_w eine 1er Liste anhängen
            # print("len(self.u_w):", len(self.u_w))
            list[0] = len(self.u_w)  # erstes Listen-Objekt wird durch die Dimensionszahl von u_w ersetzt
            # print("self.u_w.view(list):", self.u_w.view(list))
            u_w = self.u_w.view(list).expand_as(u)
            # print("list:", list)
            # print("u_w.shape:", self.u_w.shape)
            # u_w = self.u_w.view(list)
            # print("u_w.shape:", self.u_w.shape)
            # [3, 80, 60] -> [3, 120, 80, 60]

        rho_self = (1 / (1 - u_w[np.argwhere(self.direction != 0).item()]
                         * self.lattice.e[self.velocities_in[0], np.argwhere(self.direction != 0).item()])
                    * (torch.sum(f[[np.setdiff1d(np.arange(self.lattice.Q), [self.velocities_in, self.velocities_out])]
                                   + self.index] + 2 * f[[self.velocities_out] + self.index], dim=0)))

        filter_density = True  # ginge auch über Tc=0 quasi false zu setzen!
        # in Vortests war die Boundary ohne Filter stabiler!
        if filter_density == True:
            # density filtering as proposed by https://www.researchgate.net/publication/257389374_Computational_Gas_Dynamics_with_the_Lattice_Boltzmann_Method_Preconditioning_and_Boundary_Conditions
            rho_w = (rho_self + Tc * self.rho_old) / (1 + Tc)
            self.rho_old = rho_w
        else:
            rho_w = rho_self
            self.rho_old = rho_w
        # print("rho_w.shape:", rho_w.shape)
        # print("u_w.shape:", u_w.shape)
        # print("f[other].shape:", f[other].shape)
        # print("rho.shape", rho.shape)
        # print("rho_self.shape:", rho_self.shape)
        # print("u.shape", u.shape)

        ## f[here] = self.lattice.equilibrium(rho_w, u_w) + (f[other] - self.lattice.equilibrium(rho, u))  ## EQLM ist anders mit torch.einsum bzw. lattice.einsum definiert... bruh // hier spielt mir die Definition von lettuce.einsum in die Quere, zwischen Martins branch und dem aktuellen Lettuce!
        f[here] = (torch.einsum("q,q...->q...", self.lattice.w, (rho_w * (
                    (2 * torch.tensordot(self.lattice.e, u_w, dims=1) - torch.einsum("d...,d...->...", u_w, u_w)) / (
                        2 * self.lattice.cs ** 2) + 0.5 * (
                                torch.tensordot(self.lattice.e, u_w, dims=1) / (self.lattice.cs ** 2)) ** 2 + 1))) +
                   (f[other] - torch.einsum("q,q...->q...", self.lattice.w, (rho * ((2 * torch.tensordot(self.lattice.e,
                                                                                                         u,
                                                                                                         dims=1) - torch.einsum(
                       "d...,d...->...", u, u)) / (2 * self.lattice.cs ** 2) + 0.5 * (torch.tensordot(self.lattice.e, u,
                                                                                                      dims=1) / (
                                                                                                  self.lattice.cs ** 2)) ** 2 + 1)))
                    )
                   )

        ## >>> VON MIR FALSCH IN AKTUELLES LETTUCE ÜBERSETZT: HIER FEHLTE DAS "_w" AN U UND RHO IM ERSTEN EQUILIBRIUM!
        # f[here] = (torch.einsum("q,q...->q...",self.lattice.w, (rho * ((2 * torch.tensordot(self.lattice.e, u, dims=1) - torch.einsum("d...,d...->...", u, u)) / (2 * self.lattice.cs ** 2) + 0.5 * (torch.tensordot(self.lattice.e, u, dims=1) / (self.lattice.cs ** 2)) ** 2 + 1)))
        #            + (f[other] - torch.einsum("q,q...->q...", self.lattice.w, (rho * ((2 * torch.tensordot(self.lattice.e, u, dims=1) - torch.einsum("d...,d...->...", u, u)) / (2 * self.lattice.cs ** 2) + 0.5 * (torch.tensordot(self.lattice.e, u, dims=1) / (self.lattice.cs ** 2)) ** 2 + 1)))))
        ## <<<
        return f

    nsm_type = 'all'  # 'all', 'q_index', 'None',...
    # Welche relevanten "Arten" Populationen gibt's
    #   - alle, die eine positive x-Komponente haben (velocities_in)  // sollten sowieso überschrieben werden, haben aber im Zweifel für die Berechnung von rho, rho_old, rho_self einen Einfluss?
    #   - alle, die eine negative x-Komponente haben (velocities_out)  // sollten höchstens einen Einfluss auf die Berechnung von rho, rho_old, rho_self haben...
    #   - alle, ohne x-Komponente ("velocities_orthogonal" existiert noch nicht)  // sollten höchstens "Unschärfe" reinbringen...
    #   - alle, die "nur" eine positive oder negative x-Komponente haben ? <- ergibt wenig Sinn

    if nsm_type == 'all':  # alle Pops auf der Inlet-Ebene
        # "hardcore" Variante
        # war in Vortests zusammen mit q_index_in die zweitstabilste in Kombination mit q_index_in
        def make_no_stream_mask(self, f_shape):
            no_stream_mask = torch.zeros(size=f_shape, dtype=torch.bool, device=self.lattice.device)
            no_stream_mask[:, 0, :, :] = 1  # alle Pop auf der ersten Ebene
            return no_stream_mask
    elif nsm_type == 'q_index_in':  # nur velocities_in+index (s.u.) // ZeroGradientOutlet, KineticBoudnaryOutlet, ConvectiveBoundaryOutlet in MK/CD/lettuce Branch (dort ohne "_in")
        # entspricht auch dem auskommentierten im lettuceMPI_new Branch
        ## (!) war in Vortests eine der stabilsten Varianten in Kombination OHNE NCM
        def make_no_stream_mask(self, f_shape):
            no_stream_mask = torch.zeros(size=f_shape, dtype=torch.bool, device=self.lattice.device)
            no_stream_mask[[self.velocities_in] + self.index] = 1
            return no_stream_mask
    # TESTS
    elif nsm_type == 'ABB_outlet_in':
        def make_no_stream_mask(self, f_shape):
            no_stream_mask = torch.zeros(size=f_shape, dtype=torch.bool, device=self.lattice.device)
            no_stream_mask[[np.array(self.lattice.stencil.opposite)[self.velocities_in]] + self.index] = 1
            return no_stream_mask
    elif nsm_type == 'ABB_outlet_out':
        def make_no_stream_mask(self, f_shape):
            no_stream_mask = torch.zeros(size=f_shape, dtype=torch.bool, device=self.lattice.device)
            no_stream_mask[[np.array(self.lattice.stencil.opposite)[self.velocities_out]] + self.index] = 1
            return no_stream_mask
    elif nsm_type == 'EQ_outlet_P_in':
        def make_no_stream_mask(self, f_shape):
            no_stream_mask = torch.zeros(size=f_shape, dtype=torch.bool, device=self.lattice.device)
            no_stream_mask[[np.setdiff1d(np.arange(self.lattice.Q), self.velocities_in)] + self.index] = 1
            return no_stream_mask
    elif nsm_type == 'EQ_outlet_P_out':
        def make_no_stream_mask(self, f_shape):
            no_stream_mask = torch.zeros(size=f_shape, dtype=torch.bool, device=self.lattice.device)
            no_stream_mask[[np.setdiff1d(np.arange(self.lattice.Q), self.velocities_out)] + self.index] = 1
            return no_stream_mask
    elif nsm_type == 'q_index_out':
        def make_no_stream_mask(self, f_shape):
            no_stream_mask = torch.zeros(size=f_shape, dtype=torch.bool, device=self.lattice.device)
            no_stream_mask[[self.velocities_out] + self.index] = 1
            return no_stream_mask

    elif nsm_type == 'SEI':
        def make_no_stream_mask(self, f_shape):
            no_stream_mask = torch.zeros(size=f_shape, dtype=torch.bool, device=self.lattice.device)
            no_stream_mask[
                [np.concatenate(np.argwhere(np.matmul(self.lattice.stencil.e, [-1, 0, 0]) < -1 + 1e-6), axis=0)] + [0,
                                                                                                                    ...]] = 1  # entspricht bis auf das letzte [0,...] quasi dem, was für die velocities_in für die NEEQInlet rauskommt...
            return no_stream_mask

    ncm_type = False  # 'True', 'False', Collision seams to stabilize the boundary...
    # collision könnte einen abschwächenden Effekt auf die Boundary haben.
    if ncm_type:  # toggle no_collision_mask usage
        def make_no_collision_mask(self, f_shape):
            no_collision_mask = torch.zeros(size=f_shape[1:], dtype=torch.bool, device=self.lattice.device)
            no_collision_mask[self.index] = 1
            return no_collision_mask


class SyntheticEddyInlet(object):
    # according to description in https://doi.org/10.1016/j.jweia.2021.104560 ONLY 3D only in x direction so far

    # NOTES: isotropic fluctuating field is produced on the inlet plane
    # as sum of different made up vorteces

    # fluctuating velocity field that fits prescribed reynolds stree tensor is computed using Cholesky decomposition

    # isotropic gaussian shape function does something?!

    # dann u mit freeflow windspeed addiert

    def __init__(self, lattice, units, grid, L, K, N, R, rho, u_0, velocityProfile, direction=[1, 0, 0]):
        self.lattice = lattice
        self.units = units
        self.grid = grid
        self.rho = rho
        self.L = L
        self.reynolds_stress_tensor = R
        self.u_0 = u_0
        self.K = K
        self.N = N  # TODO mal N pro Fläche bei denen und bei mir ausrechnen, ob das passt
        self.direction = direction
        self.velocityProfile = velocityProfile

        # produce N random vorteces 2L downstream from inlet
        self.vorteces = torch.zeros((N, 6), device=self.lattice.device)
        self.vorteces[:, 0] = torch.rand(N, device=self.lattice.device) * -2 * self.L
        for i in range(1, 3):
            self.vorteces[:, i] = torch.rand(N, device=self.lattice.device) * self.units.convert_length_to_pu(
                torch.tensor(self.grid.shape, device=self.lattice.device))[i]
        self.vorteces[:, 3:6] = torch.rand((N, 3), device=self.lattice.device) - 0.5

        # Cholesky decomposition of prescribed Reynolds stress tensor
        R = self.reynolds_stress_tensor(self.lattice.convert_to_tensor(self.grid()[2][0, :, :]), self.u_0)
        A = torch.zeros_like(R)
        A[..., 0, 0] = torch.sqrt(R[..., 0, 0])
        A[..., 1, 0] = R[..., 1, 0] / A[..., 0, 0]
        A[..., 1, 1] = torch.sqrt(R[..., 1, 1] - A[..., 1, 0] ** 2)
        A[..., 2, 0] = R[..., 2, 0] / A[..., 0, 0]
        A[..., 2, 1] = (R[..., 2, 1] - A[..., 1, 0] * A[..., 2, 0]) / A[..., 1, 1]
        A[..., 2, 2] = torch.sqrt(R[..., 2, 2] - A[..., 2, 0] ** 2 - A[..., 2, 1] ** 2)
        print(f"Redacted Nans: {torch.isnan(A.view(-1)).sum().item()}")
        A = torch.nan_to_num(A, nan=0)
        self.A = A * self.K

        grid = self.lattice.convert_to_tensor(self.grid())
        self.grid_extended = torch.cat((torch.flip(grid[:, 1:4, ...], [1]), grid), dim=1)
        self.grid_extended[0, 0:3, ...] = self.grid_extended[0, 0:3, ...] * -1

    def hasTrueEntrys(self):
        return True

    def __call__(self, f):
        # move vorteces passively at each time step by Uĉ until they pass the inlet... each time a vortex passes the inlet a new one is produced at x - L
        self.vorteces[:, 0] += self.velocityProfile(self.vorteces[:, 2], self.u_0) * self.units.convert_time_to_pu(
            1)  # TODO freeflow windspeed einsetzen (was ist das?)? EXPERIMENTELL: u(z) statt 0.8 * self.u_0
        replace = torch.where((self.vorteces[:, 0] > self.L))[0].tolist()
        if len(replace) > 0:
            self.vorteces[[replace] + [0]] = -self.L
            self.vorteces[[replace], slice(1, 3)] = torch.rand([len(replace), 2],
                                                               device=self.lattice.device) * self.units.convert_length_to_pu(
                torch.tensor(self.grid.shape, device=self.lattice.device))[1:3]
            self.vorteces[[replace], slice(3, 6)] = torch.rand([len(replace), 3], device=self.lattice.device) - 0.5

        def shape_fun(x, sigma=0.225):
            return 2 * torch.exp(-1 / (2 * sigma ** 2) * x ** 2)

        def calculate_f(u, rho):
            """Initialize the distribution function values. The f^(1) contributions are approximated by finite differences.
            See Krüger et al. (2017).
            """

            grad_u0 = torch_gradient(u[0], dx=1, order=6)[None, ...]

            grad_u1 = torch_gradient(u[1], dx=1, order=6)[None, ...]
            S = torch.cat([grad_u0, grad_u1])

            if self.lattice.D == 3:
                grad_u2 = torch_gradient(u[2], dx=1, order=6)[None, ...]
                S = torch.cat([S, grad_u2])

            Pi_1 = 1.0 * self.units.relaxation_parameter_lu * rho * S / self.lattice.cs ** 2

            Q = (torch.einsum('ia,ib->iab', self.lattice.e, self.lattice.e)
                 - torch.eye(self.lattice.D, device=self.lattice.device,
                             dtype=self.lattice.dtype) * self.lattice.cs ** 2)

            Pi_1_Q = torch.einsum('ab...,iab->i...', Pi_1, Q)
            fneq = torch.einsum('i,i...->i...', self.lattice.w, Pi_1_Q)[:, 3, ...]
            feq = self.lattice.equilibrium(rho[3, ...], u[:, 3, ...])
            return feq - fneq

        u = torch.einsum("xyzN, Nj -> jxyz", (
                    shape_fun((self.grid_extended[0][0:4, :, :, None] - self.vorteces[:, 0]) / self.L) * (
                        shape_fun((self.grid_extended[1][0:4, :, :, None] - self.vorteces[:, 1]) / self.L) + shape_fun((
                                                                                                                                   self.grid_extended[
                                                                                                                                       1][
                                                                                                                                   0:4,
                                                                                                                                   :,
                                                                                                                                   :,
                                                                                                                                   None] - self.vorteces[
                                                                                                                                           :,
                                                                                                                                           1] +
                                                                                                                                   self.grid_extended[
                                                                                                                                       1][
                                                                                                                                       4, -1, -1]) / self.L) + shape_fun(
                    (self.grid_extended[1][0:4, :, :, None] - self.vorteces[:, 1] - self.grid_extended[1][
                        4, -1, -1]) / self.L)) * shape_fun(
                (self.grid_extended[2][0:4, :, :, None] - self.vorteces[:, 2]) / self.L)),
                         torch.sign(self.vorteces[:, 3:6])) / np.sqrt(self.N)
        u = torch.einsum('...ij, j... -> i...', self.A, u)
        u[0] += self.velocityProfile(self.grid_extended[2][0:4, :, :], self.u_0)
        u = torch.cat((self.units.convert_velocity_to_lu(u), self.lattice.u(f[:, 1:4, ...])), dim=1)
        rho = torch.ones_like(
            u[0]) * self.rho  # TODO ersetzt anderes rho, weil ja p addiert werden sollte und nicht rho?
        # rho = self.units.convert_density_to_pu(self.units.convert_pressure_pu_to_density_lu(0.5 * self.rho * torch.norm(u, dim=0) ** 2))
        # calculate feq and fneq from u and rho
        f[:, 0, :, 1:] = calculate_f(u, self.units.convert_density_to_lu(rho))[..., 1:]
        return f

    def make_no_stream_mask(self, f_shape):
        no_stream_mask = torch.zeros(size=f_shape, dtype=torch.bool, device=self.lattice.device)
        no_stream_mask[
            [np.concatenate(np.argwhere(np.matmul(self.lattice.stencil.e, [-1, 0, 0]) < -1 + 1e-6), axis=0)] + [0,
                                                                                                                ...]] = 1
        return no_stream_mask