"""
Boundary Conditions.

The `__call__` function of a boundary defines its application to the distribution functions.

Boundary conditions can define a mask (a boolean numpy array)
that specifies the grid points on which the boundary
condition operates.

Boundary classes can define two functions `make_no_stream_mask` and `make_no_collision_mask`
that prevent streaming and collisions on the boundary nodes.

The no-stream mask has the same dimensions as the distribution functions (Q, x, y, (z)) .
The no-collision mask has the same dimensions as the grid (x, y, (z)).

"""

import torch
import numpy as np
from lettuce import (LettuceException)

__all__ = ["BounceBackBoundary", "HalfwayBounceBackBoundary", "FullwayBounceBackBoundary", "AntiBounceBackOutlet", "EquilibriumBoundaryPU", "EquilibriumInletPU", "EquilibriumOutletP"]


class BounceBackBoundary:
    """Fullway Bounce-Back Boundary"""

    def __init__(self, mask, lattice):
        self.mask = lattice.convert_to_tensor(mask)
        self.lattice = lattice

    def __call__(self, f):
        f = torch.where(self.mask, f[self.lattice.stencil.opposite], f)
        return f

    def make_no_collision_mask(self, f_shape):
        assert self.mask.shape == f_shape[1:]
        return self.mask


class FullwayBounceBackBoundary:
    """Fullway Bounce-Back Boundary
    - inverts populations within two substeps
    - call() must be called after Streaming substep
    - calc_force_on_boundary() must be called after collision substep and before streaming substep
    """
    # based on Master-Branch "class BounceBackBoundary"
    # added option to calculate force on the boundary by Momentum Exchange Method

    def __init__(self, mask, lattice):
        self.mask = lattice.convert_to_tensor(mask)
        self.lattice = lattice
        self.force = torch.zeros_like(self.lattice.convert_to_tensor(self.lattice.stencil.e[0]))  # force in all D dimensions (x,y,(z))
        # create f_mask, needed for force-calculation
        # ...(marks all fs which point from fluid to solid (boundary))
        if self.lattice.D == 2:
            nx, ny = mask.shape  # Anzahl x-Punkte, Anzahl y-Punkte (Skalar), (der gesamten Simulationsdomain)
            self.f_mask = np.zeros((self.lattice.Q, nx, ny), dtype=bool)
                # f_mask: [stencilVektor-Zahl, nx, ny], Markierung aller Populationen, die im nächsten Streaming von Fluidknoten auf Boundary-Knoten strömen
                # ...zur markierung aller auf die Boundary (bzw. das Objekt, die Wand) zeigenden Stencil-Vektoren bzw. Populationen
            a, b = np.where(mask)
                # np.array: Liste der (a) x-Koordinaten  und (b) y-Koordinaten der boundary-mask
                # ...um über alle Boundary/Objekt/Wand-Knoten iterieren zu können
            for p in range(0, len(a)):  # für alle TRUE-Punkte der boundary-mask
                for i in range(0, self.lattice.Q):  # für alle stencil-Richtungen c_i (hier lattice.stencil.e)
                    try:  # try in case the neighboring cell does not exist (an f pointing out of the simulation domain)
                        if not mask[a[p] + self.lattice.stencil.e[i, 0], b[p] + self.lattice.stencil.e[i, 1]]:
                            # falls in einer Richtung Punkt+(e_x, e_y; e ist c_i) False ist, ist das also ein Oberflächepunkt des Objekts (selbst True mit Nachbar False)
                            # ...wird der an diesem Fluidknoten antiparallel dazu liegende Stencil-Vektor markiert:
                            # markiere alle "zur Boundary zeigenden" Populationen im Fluid-Bereich (also den unmittelbaren Nachbarknoten der Boundary)
                            self.f_mask[self.lattice.stencil.opposite[i], a[p] + self.lattice.stencil.e[i, 0], b[p] + self.lattice.stencil.e[i, 1]] = 1
                            # f_mask[q,x,y]
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there
        if self.lattice.D == 3:  # entspricht 2D, nur halt in 3D...guess what...
            nx, ny, z = mask.shape
            self.f_mask = np.zeros((self.lattice.Q, nx, ny, z), dtype=bool)
            a, b, c = np.where(mask)
            for p in range(0, len(a)):
                for i in range(0, self.lattice.Q):
                    try:  # try in case the neighboring cell does not exist (an f pointing out of simulation domain)
                        if not mask[a[p] + self.lattice.stencil.e[i, 0], b[p] + self.lattice.stencil.e[i, 1], c[p] + self.lattice.stencil.e[i, 2]]:
                            self.f_mask[self.lattice.stencil.opposite[i], a[p] + self.lattice.stencil.e[i, 0], b[p] + self.lattice.stencil.e[i, 1], c[p] + self.lattice.stencil.e[i, 2]] = 1
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there
        self.f_mask = self.lattice.convert_to_tensor(self.f_mask)

    def __call__(self, f):
        # FULLWAY-BB: inverts populations on all boundary nodes
        f = torch.where(self.mask, f[self.lattice.stencil.opposite], f)
        return f

    def make_no_collision_mask(self, f_shape):
        assert self.mask.shape == f_shape[1:]
        return self.mask

    def calc_force_on_boundary(self, f):
        # calculate force on boundary by momentum exchange method (MEA, MEM):
            # momentum (f_i*c_i - f_i_opposite*c_i_opposite = 2*f_i*c_i for a resting boundary)is summed for all...
            # ...populations pointing at the surface of the boundary
        tmp = torch.where(self.f_mask, f, torch.zeros_like(f))  # alle Populationen f, welche auf die Boundary zeigen
        #self.force = 1 ** self.lattice.D * 2 * torch.einsum('i..., id -> d', tmp, self.lattice.e) / 1.0  # BERECHNET KRAFT / v1.1 - M.Kliemank
        #self.force = dx ** self.lattice.D * 2 * torch.einsum('i..., id -> d', tmp, self.lattice.e) / dx  # BERECHNET KRAFT / v.1.2 - M.Bille (allgemeine dx und dt=dx, dx als Funktionsparameter, wurde in simulaton.py übergeben)
        self.force = 2 * torch.einsum('i..., id -> d', tmp, self.lattice.e)  # BERECHNE KRAFT / v2.0 - M.Bille: dx_lu = dt_lu ist immer 1 (!). Vermeide hier unnötige mini_Rechenoperationen etc.
            # summiert alle Kräfte in x und in y (und z) Richtung auf,
            # tmp: f an allen Stellen, die in f_mask markiert sind
            # tmp: 9 x nx x ny
            # self.lattice.e: 9 x 2 (2D) bzw. 9 x 3 (3D)
            # Zuordnung der Multiplikation über die 9 Einheitsvektoren (Richtungen, indexname i)
            # Vorzeichen kommt über die Koordianten der Stencil-Einheitsvektoren (e[0 bis 8])
            # übrig bleiben nur zwei (drei) Koordinatenrichtungen (indexname d)
            # "dx**self-lattice.D" = dx³ (3D) bzw. dx² (2D) als Vorfaktor, welcher einheitenmäßig aus Impulsdichte einen Impuls macht
                # eigentlich rechnet man hier einen DELTA P aus
                # unter Annahme des stetigen Impulsaustauschs über dt, kann die Kraft als F= dP/dt berechnet werden
                # ...deshalb wird hier nochmal durch dt=dx geteilt (weil c_i=1=dx/dt=1 kann das aber augelassen werden (v2.0) !)
        return self.force  # force in x and y (and z) direction


class HalfwayBounceBackBoundary:
    """Halfway Bounce Back Boundary
    - inverts populations within one substep
    - call() must be called after Streaming substep
    - calc_force_on_boundary() must be called after collision substep and before streaming substep
    """

    def __init__(self, mask, lattice):
        self.mask = lattice.convert_to_tensor(mask)
        self.lattice = lattice  # das self wird hier benötigt, da auf lattice auch außerhalb der init zugegriffen werden können soll
        self.force = torch.zeros_like(self.lattice.convert_to_tensor(self.lattice.stencil.e[0]))  # force in all D dimensions (x,y,(z))
        # create f_mask, needed for force-calculation
        # ...(marks all fs which point from fluid to solid (boundary))
        if self.lattice.D == 2:
            nx, ny = mask.shape  # Anzahl x-Punkte, Anzahl y-Punkte (Skalar), (der gesamten Simulationsdomain)
            self.f_mask = np.zeros((self.lattice.Q, nx, ny), dtype=bool)  # f_mask: [stencilVektor-Zahl, nx, ny], Markierung aller Populationen, die im nächsten Streaming von Fluidknoten auf Boundary-Knoten strömen
                # ...zur markierung aller auf die Boundary (bzw. das Objekt, die Wand) zeigenden Stencil-Vektoren bzw. Populationen
            a, b = np.where(mask)  # np.array: Liste der (a) x-Koordinaten  und (b) y-Koordinaten der boundary-mask
                # ...um über alle Boundary/Objekt/Wand-Knoten iterieren zu können
            for p in range(0, len(a)):  # für alle TRUE-Punkte der boundary-mask
                for i in range(0, self.lattice.Q):  # für alle stencil-Richtungen c_i (hier lattice.stencil.e)
                    try:  # try in case the neighboring cell does not exist (an f pointing out of the simulation domain)
                        if not mask[a[p] + self.lattice.stencil.e[i, 0], b[p] + self.lattice.stencil.e[i, 1]]:
                            # falls in einer Richtung Punkt+(e_x, e_y; e ist c_i) False ist, ist das also ein Oberflächepunkt des Objekts (selbst True mit Nachbar False)
                            # ...wird der an diesem Fluidknoten antiparallel dazu liegende Stencil-Vektor markiert:
                            # markiere alle "zur Boundary zeigenden" Populationen im Fluid-Bereich (also den unmittelbaren Nachbarknoten der Boundary)
                            self.f_mask[self.lattice.stencil.opposite[i], a[p] + self.lattice.stencil.e[i, 0], b[p] + self.lattice.stencil.e[i, 1]] = 1
                            # f_mask[q,x,y]
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there
        if self.lattice.D == 3:  # entspricht 2D, nur halt in 3D...guess what...
            nx, ny, z = mask.shape
            self.f_mask = np.zeros((self.lattice.Q, nx, ny, z), dtype=bool)
            a, b, c = np.where(mask)
            for p in range(0, len(a)):
                for i in range(0, self.lattice.Q):
                    try:  # try in case the neighboring cell does not exist (an f pointing out of simulation domain)
                        if not mask[a[p] + self.lattice.stencil.e[i, 0], b[p] + self.lattice.stencil.e[i, 1], c[p] + self.lattice.stencil.e[i, 2]]:
                            self.f_mask[self.lattice.stencil.opposite[i], a[p] + self.lattice.stencil.e[i, 0], b[p] + self.lattice.stencil.e[i, 1], c[p] + self.lattice.stencil.e[i, 2]] = 1
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there
        self.f_mask = self.lattice.convert_to_tensor(self.f_mask)

    def __call__(self, f, f_collided):
        # HALFWAY-BB: overwrite all populations (on fluid nodes) which came from boundary with pre-streaming populations (on fluid nodes) which pointed at boundary
            #print("f_mask:\n", self.f_mask)
            #print("f_mask(q2,x1,y1):\n", self.f_mask[2, 1, 1])
            #print("f_mask(q2,x1,y3):\n", self.f_mask[2, 1, 3])
            #print("f_mask(opposite):\n", self.f_mask[self.lattice.stencil.opposite])
        f = torch.where(self.f_mask[self.lattice.stencil.opposite], f_collided[self.lattice.stencil.opposite], f)  # ersetze alle "von der boundary kommenden" Populationen durch ihre post-collision_pre-streaming entgegengesetzten Populationen
            # ...bounce-t die post_collision/pre-streaming Populationen an der Boundary innerhalb eines Zeitschrittes
            # ...von außen betrachtet wird "während des streamings", innerhalb des gleichen Zeitschritts invertiert.
            # es wird keine no_streaming_mask benötigt, da sowieso alles, was aus der boundary geströmt käme hier durch pre-Streaming Populationen überschrieben wird.
        return f

    def make_no_stream_mask(self, f_shape):
        # ?? no_stream_mask = torch.zeros(size=f_shape, dtype=torch.bool, device=self.lattice.device)
        assert self.mask.shape == f_shape[1:]  # all dimensions except the 0th (q)
        return self.mask  #| self.mask

    def make_no_collision_mask(self, f_shape):
        assert self.mask.shape == f_shape[1:]
        return self.mask

    def calc_force_on_boundary(self, f):
        # calculate force on boundary by momentum exchange method (MEA, MEM):
            # momentum (f_i*c_i - f_i_opposite*c_i_opposite = 2*f_i*c_i for a resting boundary)is summed for all...
            # ...populations pointing at the surface of the boundary
        tmp = torch.where(self.f_mask, f, torch.zeros_like(f))  # alle Populationen f, welche auf die Boundary zeigen
        # self.force = 1 ** self.lattice.D * 2 * torch.einsum('i..., id -> d', tmp, self.lattice.e) / 1.0  # BERECHNET KRAFT / v1.1 - M.Kliemank
        # self.force = dx ** self.lattice.D * 2 * torch.einsum('i..., id -> d', tmp, self.lattice.e) / dx  # BERECHNET KRAFT / v.1.2 - M.Bille (allgemeine dx und dt=dx, dx als Funktionsparameter, wurde in simulaton.py übergeben)
        self.force = 2 * torch.einsum('i..., id -> d', tmp, self.lattice.e)  # BERECHNE KRAFT / v2.0 - M.Bille: dx_lu = dt_lu ist immer 1 (!). Vermeide hier unnötige mini_Rechenoperationen etc.
        # summiert alle Kräfte in x und in y (und z) Richtung auf,
        # tmp: f an allen Stellen, die in f_mask markiert sind
        # tmp: 9 x nx x ny
        # self.lattice.e: 9 x 2 (2D) bzw. 9 x 3 (3D)
        # Zuordnung der Multiplikation über die 9 Einheitsvektoren (Richtungen, indexname i)
        # Vorzeichen kommt über die Koordianten der Stencil-Einheitsvektoren (e[0 bis 8])
        # übrig bleiben nur zwei (drei) Koordinatenrichtungen (indexname d)
        # "dx**self-lattice.D" = dx³ (3D) bzw. dx² (2D) als Vorfaktor, welcher einheitenmäßig aus Impulsdichte einen Impuls macht
        # eigentlich rechnet man hier einen DELTA P aus
        # unter Annahme des stetigen Impulsaustauschs über dt, kann die Kraft als F= dP/dt berechnet werden
        # ...deshalb wird hier nochmal durch dt=dx geteilt (weil c_i=1=dx/dt=1 kann das aber augelassen werden (v2.0) !)
        return self.force  # force in x and y direction


class EquilibriumBoundaryPU:
    """Sets distributions on this boundary to equilibrium with predefined velocity and pressure.
    Note that this behavior is generally not compatible with the Navier-Stokes equations.
    This boundary condition should only be used if no better options are available.
    """

    def __init__(self, mask, lattice, units, velocity, pressure=0):
        # parameter input (u, p) in PU!
        self.mask = lattice.convert_to_tensor(mask)
        self.lattice = lattice
        self.units = units
        self.velocity = lattice.convert_to_tensor(velocity)
        self.pressure = lattice.convert_to_tensor(pressure)

    def __call__(self, f):
        # convert PU-inputs to LU, calc feq and overwrite f with feq where mask==True
        rho = self.units.convert_pressure_pu_to_density_lu(self.pressure)
        u = self.units.convert_velocity_to_lu(self.velocity)
        feq = self.lattice.equilibrium(rho, u)
        feq = self.lattice.einsum("q,q->q", [feq, torch.ones_like(f)])
        f = torch.where(self.mask, feq, f)

      #  ny = f.shape[2]
      #  u1 = (-1.0 * np.arange(0, ny) * (np.arange(0, ny) - ny) * self.units.characteristic_velocity_pu * 1/(ny/2)**2)[None,...]
      #  u2 = np.zeros_like(u1)
      #  u = np.stack([u1, u2], axis=0)
      #  u = self.lattice.convert_to_tensor(u)
      #  feq = self.lattice.equilibrium(rho, u)
      #  f[:, 0, :] = feq[:, 0, :]
        return f


class EquilibriumInletPU:
    """Sets distributions on this boundary to equilibrium with predefined velocity and pressure.
    Note that this behavior is generally not compatible with the Navier-Stokes equations.
    This boundary condition should only be used if no better options are available.
    """

    def __init__(self, mask, lattice, units, velocity, pressure=0):
        # parameter input (u, p) in PU!
        self.mask = lattice.convert_to_tensor(mask)
        self.lattice = lattice
        self.units = units
        self.velocity = lattice.convert_to_tensor(velocity)  # inlet-velocity in PU
        self.pressure = lattice.convert_to_tensor(pressure)  # inlet-pressure in PU
        self.u_inlet = self.units.convert_velocity_to_lu(self.velocity) # ein Tensor: tensor([0.0289, 0.0000], device='cuda:0')
        # calculate uniform or parabolic inlet-velocity-distibution
        u_in_parabel = True
        if u_in_parabel:
            ## Parabelförmige Geschwindigkeit, vom zweiten bis vorletzten Randpunkt (keine Interferenz mit lateralen Wänden (BBB  oder periodic))
                ## How to Parabel:
                ## 1.Parabel in Nullstellenform: y = (x-x1)*(x-x2)
                ## 2.nach oben geöffnete Parabel mit Nullstelle bei x1=0 und x2=x0: y=-x*(x-x0)
                ## 3.skaliere Parabel, sodass der Scheitelpunkt immer bei ys=1.0 ist: y=-x*(x-x0)*(1/(x0/2)²)
                ## (4. optional) skaliere Amplitude der Parabel mit 1.5, um dem Integral einer homogenen Einstromgeschwindigkeit zu entsprechen
            ny = mask.shape[1]  # Gitterpunktzahl in y-Richtung
            ux_temp = np.zeros((1, ny))  # x-Geschwindigkeiten der Randbedingung
            y_coordinates = np.linspace(0, ny, ny)  # linspace() erzeugt n Punkte zwischen 0 und ny inklusive 0 und ny, so wird die Parabel auch symmetrisch und ist trotzdem an den Rändern NULL
            ux_temp[:, 1:-1] = - np.array(self.u_inlet.cpu()).max() * y_coordinates[1:-1] * (y_coordinates[1:-1] - ny) * 1/(ny/2)**2
                # (!) es muss die charakteristische Geschwindigkeit in LU genutzt werden (!) -> der Unterschied PU/LU ist u.U. Größenordnungen und es kommt bei falscher Nutzung zu Schock/Überschall und somit Sim-Crash
                # Skalierungsfaktor 3/2=1.5 für die Parabelamplitude, falls man im Integral der Konstantgeschwindigkeit entsprechenn möchte.
                # in 2D braucht u1 dann die Dimension 1 x ny (!)
            uy_temp = np.zeros_like(ux_temp)  # y-Geschwindigkeit = 0
            self.u_inlet = np.stack([ux_temp, uy_temp], axis=0)  # verpacke u-Feld
            self.u_inlet = self.lattice.convert_to_tensor(self.u_inlet)  # np.array to torch.tensor

    def __call__(self, f):
        # convert PU-inputs to LU, calc feq and overwrite f with feq where mask==True
        rho = self.units.convert_pressure_pu_to_density_lu(self.pressure)
        feq = self.lattice.equilibrium(rho, self.u_inlet)  # Berechne Gleichgewicht mit neuer Geschwindigkeit
        feq = self.lattice.einsum("q,q->q", [feq, torch.ones_like(f)])  # erweitere auf komplettes Feld, falls nötig (feq "breit" ziehen in x-Richtung)
        f = torch.where(self.mask, feq, f)  # überschreibe f am Einlass mit feq
        return f


class AntiBounceBackOutlet:
    """Allows distributions to leave domain unobstructed through this boundary.
        Based on equations from page 195 of "The lattice Boltzmann method" (2016 by Krüger et al.)
        Give the side of the domain with the boundary as list [x, y, z] with only one entry nonzero
        [1, 0, 0] for positive x-direction in 3D; [1, 0] for the same in 2D
        [0, -1, 0] is negative y-direction in 3D; [0, -1] for the same in 2D
        """

    def __init__(self, lattice, direction):

        assert isinstance(direction, list), \
            LettuceException(
                f"Invalid direction parameter. Expected direction of type list but got {type(direction)}.")

        assert len(direction) in [1, 2, 3], \
            LettuceException(
                f"Invalid direction parameter. Expected direction of of length 1, 2 or 3 but got {len(direction)}.")

        assert (direction.count(0) == (len(direction) - 1)) and ((1 in direction) ^ (-1 in direction)), \
            LettuceException(
                "Invalid direction parameter. "
                f"Expected direction with all entries 0 except one 1 or -1 but got {direction}.")

        direction = np.array(direction)
        self.lattice = lattice

        # select velocities to be bounced (the ones pointing in "direction")
        self.velocities = np.concatenate(np.argwhere(np.matmul(self.lattice.stencil.e, direction) > 1 - 1e-6), axis=0)

        # build indices of u and f that determine the side of the domain
        self.index = []
        self.neighbor = []
        for i in direction:
            if i == 0:
                self.index.append(slice(None))
                self.neighbor.append(slice(None))
            if i == 1:
                self.index.append(-1)
                self.neighbor.append(-2)
            if i == -1:
                self.index.append(0)
                self.neighbor.append(1)
        # construct indices for einsum and get w in proper shape for the calculation in each dimension
        if len(direction) == 3:
            self.dims = 'dc, cxy -> dxy'
            self.w = self.lattice.w[self.velocities].view(1, -1).t().unsqueeze(1)
        if len(direction) == 2:
            self.dims = 'dc, cx -> dx'
            self.w = self.lattice.w[self.velocities].view(1, -1).t()
        if len(direction) == 1:
            self.dims = 'dc, c -> dc'
            self.w = self.lattice.w[self.velocities]

    def __call__(self, f):
        u = self.lattice.u(f)
        u_w = u[[slice(None)] + self.index] + 0.5 * (u[[slice(None)] + self.index] - u[[slice(None)] + self.neighbor])
        f[[np.array(self.lattice.stencil.opposite)[self.velocities]] + self.index] = (
                - f[[self.velocities] + self.index] + self.w * self.lattice.rho(f)[[slice(None)] + self.index] *
                (2 + torch.einsum(self.dims, self.lattice.e[self.velocities], u_w) ** 2 / self.lattice.cs ** 4
                 - (torch.norm(u_w, dim=0) / self.lattice.cs) ** 2)
        )
        return f

    def make_no_stream_mask(self, f_shape):
        no_stream_mask = torch.zeros(size=f_shape, dtype=torch.bool, device=self.lattice.device)
        no_stream_mask[[np.array(self.lattice.stencil.opposite)[self.velocities]] + self.index] = 1
        return no_stream_mask

    # not 100% sure about this. But collisions seem to stabilize the boundary.
    # def make_no_collision_mask(self, f_shape):
    #    no_collision_mask = torch.zeros(size=f_shape[1:], dtype=torch.bool, device=self.lattice.device)
    #    no_collision_mask[self.index] = 1
    #    return no_collision_mask


class EquilibriumOutletP(AntiBounceBackOutlet):
    """Equilibrium outlet with constant pressure.
    """

    def __init__(self, lattice, direction, rho0=1.0):
        super(EquilibriumOutletP, self).__init__(lattice, direction)
        self.rho0 = rho0

    def __call__(self, f):
        here = [slice(None)] + self.index
        other = [slice(None)] + self.neighbor
        rho = self.lattice.rho(f)
        u = self.lattice.u(f)
        rho_w = self.rho0 * torch.ones_like(rho[here])
        u_w = u[other]
        f[here] = self.lattice.equilibrium(rho_w[..., None], u_w[..., None])[..., 0]
        return f

    def make_no_stream_mask(self, f_shape):
        no_stream_mask = torch.zeros(size=f_shape, dtype=torch.bool, device=self.lattice.device)
        no_stream_mask[[np.setdiff1d(np.arange(self.lattice.Q), self.velocities)] + self.index] = 1
        return no_stream_mask

    def make_no_collision_mask(self, f_shape):
        no_collision_mask = torch.zeros(size=f_shape[1:], dtype=torch.bool, device=self.lattice.device)
        no_collision_mask[self.index] = 1
        return no_collision_mask
