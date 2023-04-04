"""
Observables.
Each observable is defined as a callable class.
The `__call__` function takes f as an argument and returns a torch tensor.
"""

import torch
import numpy as np
from lettuce.util import torch_gradient
from packaging import version

__all__ = ["Observable", "MaximumVelocity", "IncompressibleKineticEnergy", "Enstrophy", "EnergySpectrum", "Vorticity", "DragCoefficient", "LiftCoefficient", "Mass"]


class Observable:
    def __init__(self, lattice, flow):
        self.lattice = lattice
        self.flow = flow

    def __call__(self, f):
        raise NotImplementedError


class MaximumVelocity(Observable):
    """Maximum velocitiy"""

    def __call__(self, f):
        u = self.lattice.u(f)
        return self.flow.units.convert_velocity_to_pu(torch.norm(u, dim=0).max())


class IncompressibleKineticEnergy(Observable):
    """Total kinetic energy of an incompressible flow."""

    def __call__(self, f):
        dx = self.flow.units.convert_length_to_pu(1.0)
        kinE = self.flow.units.convert_incompressible_energy_to_pu(torch.sum(self.lattice.incompressible_energy(f)))
        kinE *= dx ** self.lattice.D
        return kinE


class Enstrophy(Observable):
    """The integral of the vorticity

    Notes
    -----
    The function only works for periodic domains
    """

    def __call__(self, f):
        u0 = self.flow.units.convert_velocity_to_pu(self.lattice.u(f)[0])
        u1 = self.flow.units.convert_velocity_to_pu(self.lattice.u(f)[1])
        dx = self.flow.units.convert_length_to_pu(1.0)
        grad_u0 = torch_gradient(u0, dx=dx, order=6)
        grad_u1 = torch_gradient(u1, dx=dx, order=6)
        vorticity = torch.sum((grad_u0[1] - grad_u1[0]) * (grad_u0[1] - grad_u1[0]))
        if self.lattice.D == 3:
            u2 = self.flow.units.convert_velocity_to_pu(self.lattice.u(f)[2])
            grad_u2 = torch_gradient(u2, dx=dx, order=6)
            vorticity += torch.sum(
                (grad_u2[1] - grad_u1[2]) * (grad_u2[1] - grad_u1[2])
                + ((grad_u0[2] - grad_u2[0]) * (grad_u0[2] - grad_u2[0]))
            )
        return vorticity * dx ** self.lattice.D


class EnergySpectrum(Observable):
    """The kinetic energy spectrum"""

    def __init__(self, lattice, flow):
        super(EnergySpectrum, self).__init__(lattice, flow)
        self.dx = self.flow.units.convert_length_to_pu(1.0)
        self.dimensions = self.flow.grid[0].shape
        frequencies = [self.lattice.convert_to_tensor(np.fft.fftfreq(dim, d=1 / dim)) for dim in self.dimensions]
        wavenumbers = torch.stack(torch.meshgrid(*frequencies))
        wavenorms = torch.norm(wavenumbers, dim=0)

        if self.lattice.D == 3:
            self.norm = self.dimensions[0] * np.sqrt(2 * np.pi) / self.dx ** 2
        else:
            self.norm = self.dimensions[0] / self.dx

        self.wavenumbers = torch.arange(int(torch.max(wavenorms)))
        self.wavemask = (
                (wavenorms[..., None] > self.wavenumbers.to(dtype=lattice.dtype, device=lattice.device) - 0.5) &
                (wavenorms[..., None] <= self.wavenumbers.to(dtype=lattice.dtype, device=lattice.device) + 0.5)
        )

    def __call__(self, f):
        u = self.lattice.u(f)
        return self.spectrum_from_u(u)

    def spectrum_from_u(self, u):
        u = self.flow.units.convert_velocity_to_pu(u)
        ekin = self._ekin_spectrum(u)
        ek = ekin[..., None] * self.wavemask.to(dtype=self.lattice.dtype)
        ek = ek.sum(torch.arange(self.lattice.D).tolist())
        return ek

    def _ekin_spectrum(self, u):
        """distinguish between different torch versions"""
        torch_ge_18 = (version.parse(torch.__version__) >= version.parse("1.8.0"))
        if torch_ge_18:
            return self._ekin_spectrum_torch_ge_18(u)
        else:
            return self._ekin_spectrum_torch_lt_18(u)

    def _ekin_spectrum_torch_lt_18(self, u):
        zeros = torch.zeros(self.dimensions, dtype=self.lattice.dtype, device=self.lattice.device)[..., None]
        uh = (torch.stack([
            torch.fft(torch.cat((u[i][..., None], zeros), self.lattice.D),
                      signal_ndim=self.lattice.D) for i in range(self.lattice.D)]) / self.norm)
        ekin = torch.sum(0.5 * (uh[..., 0] ** 2 + uh[..., 1] ** 2), dim=0)
        return ekin

    def _ekin_spectrum_torch_ge_18(self, u):
        uh = (torch.stack([
            torch.fft.fftn(u[i], dim=tuple(torch.arange(self.lattice.D))) for i in range(self.lattice.D)
        ]) / self.norm)
        ekin = torch.sum(0.5 * (uh.imag ** 2 + uh.real ** 2), dim=0)
        return ekin


class Mass(Observable):
    """Total mass in lattice units.

    Parameters
    ----------
    no_mass_mask : torch.Tensor
        Boolean mask that defines grid points
        which do not count into the total mass (e.g. bounce-back boundaries).
    """

    def __init__(self, lattice, flow, no_mass_mask=None):
        super(Mass, self).__init__(lattice, flow)
        self.mask = no_mass_mask

    def __call__(self, f):
        mass = f[..., 1:-1, 1:-1].sum()
        if self.mask is not None:
            mass -= (f * self.mask.to(dtype=torch.float)).sum()
        return mass


class Vorticity(Observable):
    """Vorticity: the curl of the flow velocity field
    Note: only works for periodic domains (according to a note in class Enstrophy (see above)
    """
    def __call__(self, f):
        u0 = self.flow.units.convert_velocity_to_pu(self.lattice.u(f)[0])
        u1 = self.flow.units.convert_velocity_to_pu(self.lattice.u(f)[1])
        dx = self.flow.units.convert_length_to_pu(1.0)
        grad_u0 = torch_gradient(u0, dx=dx, order=6)
        grad_u1 = torch_gradient(u1, dx=dx, order=6)
        vorticity = (grad_u0[1] - grad_u1[0]) * (grad_u0[1] - grad_u1[0])  # gegenüber Enstrophy fehlt hier die Summation
        if self.lattice.D == 3:
            u2 = self.flow.units.convert_velocity_to_pu(self.lattice.u(f)[2])
            grad_u2 = torch_gradient(u2, dx=dx, order=6)
            vorticity += (grad_u2[1] - grad_u1[2]) * (grad_u2[1] - grad_u1[2])\
                + ((grad_u0[2] - grad_u2[0]) * (grad_u0[2] - grad_u2[0]))  # gegenüber Enstrophy fehlt hier die Summation

        return vorticity * dx ** self.lattice.D


class DragCoefficient(Observable):
    """The drag coefficient of an obstacle, calculated using momentum exchange method (MEM, MEA) according to a
    modified version of M.Kliemank's Drag Coefficient Code

    calculates the density, gets the force in x direction on the obstacle boundary,
    calculates the coefficient of drag

    M.K.'s non used code is commented out by "##"
    """

    def __init__(self, lattice, flow, simulation, area):
        super().__init__(lattice, flow)
        self.forceVal = simulation.forceVal
      #  self.area_pu = area  # crosssectional area of obstacle (! length in 2D -> area-dimension = self.lattice.D-1)
        self.area_lu = area * (self.flow.units.characteristic_length_lu/self.flow.units.characteristic_length_pu) ** (self.lattice.D-1)
        #self.rho_max_list = []
        #self.rho_min_list = []
        #self.rho_mean_list = []
        ## self.lattice = lattice
        ## self.flow = flow
        ## self.boundary = []
        ## for boundary in simulation._boundaries:
        ##     if isinstance(boundary, BounceBackBoundary):
        ##         boundary.output_force = True
        ##         self.boundary.append(boundary)

    def __call__(self, f):
        #rho = torch.mean(self.lattice.rho(f[:, 0, ...]))
        rho_tmp = torch.where(self.lattice.convert_to_tensor(self.flow.solid_mask), self.lattice.convert_to_tensor(torch.nan), self.lattice.rho(f))
        rho = torch.nanmean(rho_tmp)
        #self.rho_max_list.append(self.lattice.convert_to_numpy(torch.max(rho_tmp[~rho_tmp.isnan()])))
        #self.rho_min_list.append(self.lattice.convert_to_numpy(torch.min(rho_tmp[~rho_tmp.isnan()])))
        #self.rho_mean_list.append(self.lattice.convert_to_numpy(rho))
       # rho_pu = self.flow.units.convert_density_to_pu(rho)  # what about "characteristic mass"?
        force_x_lu = self.forceVal[-1][0]
       # force_x_pu = self.flow.units.convert_force_to_pu(self.forceVal[-1][0]) # Fx ist die Kraft in x-Richtung, force sind die Kraft in x und in y-Richtung (+z in 3D)
        ## f = torch.where(self.mask, f, torch.zeros_like(f))
        ## f[0, ...] = 0
        ## Fw =  self.flow.units.convert_force_to_pu(1**self.lattice.D * self.factor * torch.einsum('ixy, id -> d', [f, self.lattice.e])[0]/1)
        #drag_coefficient = force_x_pu / (0.5 * rho * self.flow.units.characteristic_velocity_lu ** 2 * self.area_pu)  # drag_coefficient in PU
        #PU: drag_coefficient = force_x_pu / (0.5 * rho_pu * self.flow.units.characteristic_velocity_pu ** 2 * self.area_pu)  # drag_coefficient in PU
        drag_coefficient = force_x_lu / (0.5 * rho * self.flow.units.characteristic_velocity_lu ** 2 * self.area_lu)  # drag_coefficient in LU
        return drag_coefficient


class LiftCoefficient(Observable):
    """The lift coefficient of an obstacle, calculated using momentum exchange method (MEM, MEA) according to a
        modified version of M.Kliemank's lift Coefficient Code

        calculates the density, gets the force in y direction on the obstacle boundary,
        calculates the coefficient of lift

        M.K.'s non used code is commented out by "##"
        """

    def __init__(self, lattice, flow, simulation, area):
        super().__init__(lattice, flow)
        self.forceVal = simulation.forceVal
      #  self.area_pu = area  # crosssectional area of obstacle
        self.area_lu = area * (self.flow.units.characteristic_length_lu / self.flow.units.characteristic_length_pu) ** (
                    self.lattice.D - 1)
        ## self.lattice = lattice
        ## self.flow = flow
        ## self.boundary = []
        ## for boundary in simulation._boundaries:
        ##     if isinstance(boundary, BounceBackBoundary):
        ##         boundary.output_force = True
        ##         self.boundary.append(boundary)

    def __call__(self, f):
        #rho = torch.mean(self.lattice.rho(f[:, 0, ...]))
        rho_tmp = torch.where(self.lattice.convert_to_tensor(self.flow.solid_mask), self.lattice.convert_to_tensor(torch.nan),
                              self.lattice.rho(f))
        rho = torch.nanmean(rho_tmp)
      #  rho_pu = self.flow.units.convert_density_to_pu(rho)  # what about "characteristic mass"?
        force_y_lu = self.forceVal[-1][1]
      #  force_y_pu = self.flow.units.convert_force_to_pu(self.forceVal[-1][1]) # Fy ist die Kraft in y-Richtung, force sind die Kraft in x (force[0]) und in y-Richtung (force[1])
        ## f = torch.where(self.mask, f, torch.zeros_like(f))
        ## f[0, ...] = 0
        ## Fw =  self.flow.units.convert_force_to_pu(1**self.lattice.D * self.factor * torch.einsum('ixy, id -> d', [f, self.lattice.e])[0]/1)
        #lift_coefficient = force_y / (0.5 * rho * self.flow.units.characteristic_velocity_lu ** 2 * self.area)
        #lift_coefficient = force_y_pu / (0.5 * rho_pu * self.flow.units.characteristic_velocity_pu ** 2 * self.area_pu)  # lift_coefficient in PU
        lift_coefficient = force_y_lu / (0.5 * rho * self.flow.units.characteristic_velocity_lu ** 2 * self.area_lu)  # lift_coefficient in LU
        return lift_coefficient
