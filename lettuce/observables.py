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
        vorticity = (grad_u0[1] - grad_u1[0]) * (grad_u0[1] - grad_u1[0])  # fyi.: no summation in comparison to Enstrophy
        if self.lattice.D == 3:
            u2 = self.flow.units.convert_velocity_to_pu(self.lattice.u(f)[2])
            grad_u2 = torch_gradient(u2, dx=dx, order=6)
            vorticity += (grad_u2[1] - grad_u1[2]) * (grad_u2[1] - grad_u1[2])\
                + ((grad_u0[2] - grad_u2[0]) * (grad_u0[2] - grad_u2[0]))  # fyi.: no summation in comparison to Enstrophy

        return vorticity * dx ** self.lattice.D


class DragCoefficient(Observable):
    """The drag coefficient of an obstacle, calculated using momentum exchange method (MEM, MEA) according to a
    modified version of M.Kliemank's Drag Coefficient Code

    calculates the density, gets the force in x direction on the obstacle boundary,
    calculates the coefficient of drag
    """

    def __init__(self, lattice, flow, obstacle_boundary, area):
        super().__init__(lattice, flow)
        self.obstacle_boundary = obstacle_boundary
        self.area_lu = area * (self.flow.units.characteristic_length_lu/self.flow.units.characteristic_length_pu) ** (self.lattice.D-1) # crosssectional area of obstacle in LU (! lengthdimension in 2D -> area-dimension = self.lattice.D-1)
        self.nan = self.lattice.convert_to_tensor(torch.nan)
        self.solid_mask = self.lattice.convert_to_tensor(self.flow.solid_mask)

    def __call__(self, f):
        #OLD rho = torch.mean(self.lattice.rho(f[:, 0, ...]))  # simple rho_mean, including the boundary region
        # rho_mean (excluding boundary region):
        rho_tmp = torch.where(self.solid_mask, self.nan, self.lattice.rho(f))
        rho = torch.nanmean(rho_tmp)
        force_x_lu = self.obstacle_boundary.force_sum[0]  # get current force on obstacle in x direction
        drag_coefficient = force_x_lu / (0.5 * rho * self.flow.units.characteristic_velocity_lu ** 2 * self.area_lu)  # calculate drag_coefficient in LU
        return drag_coefficient


class LiftCoefficient(Observable):
    """The lift coefficient of an obstacle, calculated using momentum exchange method (MEM, MEA) according to a
        modified version of M.Kliemank's lift Coefficient Code

        calculates the density, gets the force in y direction on the obstacle boundary,
        calculates the coefficient of lift
        """

    def __init__(self, lattice, flow, obstacle_boundary, area):
        super().__init__(lattice, flow)
        self.obstacle_boundary = obstacle_boundary
        self.area_lu = area * (self.flow.units.characteristic_length_lu / self.flow.units.characteristic_length_pu) ** (
                    self.lattice.D - 1)
        self.nan = self.lattice.convert_to_tensor(torch.nan)
        self.solid_mask = self.lattice.convert_to_tensor(self.flow.solid_mask)

    def __call__(self, f):
        #OLD rho = torch.mean(self.lattice.rho(f[:, 0, ...]))  # simple rho_mean, including the boundary region
        # rho_mean (excluding boundary region):
        rho_tmp = torch.where(self.solid_mask, self.nan, self.lattice.rho(f))
        rho = torch.nanmean(rho_tmp)
        force_y_lu = self.obstacle_boundary.force_sum[1] # get current force on obstacle in y direction
        lift_coefficient = force_y_lu / (0.5 * rho * self.flow.units.characteristic_velocity_lu ** 2 * self.area_lu)  # calculate lift_coefficient in LU
        return lift_coefficient

class ForceOnBoundary(Observable):
    """Force on a bounce back boundary, calculated by momentum exchange method (see boundary.py)
        returns the force-field (individual force on nodes) and the force_sum (summed force vector on whole boundary)
        12.06.23: not tested yet
        """
    def __init__(self, lattice, flow, obstacle_boundary):
        super().__init__(lattice, flow)
        self.obstacle_boundary = obstacle_boundary

    def __call__(self, f):
        return self.obstacle_boundary.force, self.obstacle_boundary.force_sum
