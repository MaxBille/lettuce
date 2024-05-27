import matplotlib as mpl
from matplotlib import pyplot as plt
from runstats import Statistics
from lettuce.unit import UnitConversion
from lettuce.boundary import EquilibriumBoundaryPU, BounceBackBoundary, AntiBounceBackOutlet, SyntheticEddyInlet
from lettuce.grid import RegularGrid
import lettuce as lt
import numpy as np
import os
import torch
import sys
import time


class Test3D(object):
    """Flow class to simulate the flow around an object (mask) in 3D.
    See documentation for :class:`~Obstacle2D` for details.
    """

    def __init__(self, resolution_x, resolution_y, resolution_z, reynolds_number, mach_number, lattice, char_length_lu, char_length_pu, char_velocity_pu, char_density_pu, K_Factor, L, N, velocity_profile):
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.resolution_z = resolution_z
        self.K_Factor = K_Factor
        self.N = N
        self.L = L
        if velocity_profile == "ref":
            self.wind_speed_profile = self._wind_speed_profile
        else:
            self.wind_speed_profile = self._wind_speed_profile_turb
        self.mpiObject = lattice.mpiObject
        self.units = lt.UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=char_length_lu, characteristic_length_pu=char_length_pu,
            characteristic_velocity_pu=char_velocity_pu,
            characteristic_density_pu=char_density_pu, characteristic_density_lu=1
        )
        self.rgrid = lt.RegularGrid([resolution_x, resolution_y, resolution_z], self.units.characteristic_length_lu,
                                   self.units.characteristic_length_pu, endpoint=False, mpiObject=lattice.mpiObject)

        self._mask = np.zeros(shape=self.rgrid.shape, dtype=bool)

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, m):
        assert isinstance(m, np.ndarray) and m.shape == (self.resolution_x, self.resolution_y, self.resolution_z)
        self._mask = m.astype(bool)

    def initial_solution(self, x):
        p = np.zeros_like(x[0], dtype=float)[None, ...]
        u = np.zeros((len(x),) + x[0].shape)
        u[0] = np.where(self.mask, 0, self.wind_speed_profile(self.units.lattice.convert_to_tensor(x[2]), self.units.characteristic_velocity_pu).cpu().numpy())
        return p, u

    @property
    def grid(self):
        return self.rgrid()

    def _wind_speed_profile_turb(self, z, u_0):
        # based on DIN Onstwaswindprofil
        # all inputs in PU
        # return 0.77 * u_max * (z / 10) ** (alpha)
        roughness_length = z_0 = 0.02  # meter
        k_r = 0.19 * (z_0 / 0.05) ** 0.07
        z_min = 1.2

        #test =
        #plt.plot(test[0, :].cpu().numpy())
        #plt.show()
        return torch.where(z < z_min, u_0 * k_r * torch.log(torch.tensor(z_min / z_0, device=self.units.lattice.device)) * 1,
                    u_0 * k_r * torch.log(z / z_0) * 1)

    def _wind_speed_profile(self, z, u_max, alpha=0.25):
        # based on wikipedia article about wind profile
        # return u_max * (z / z_max) ** (1/alpha)
        # based on DIN Onstwaswindprofil
        # all inputs in PU
        # return 0.77 * u_max * (z / 10) ** (alpha)
        return u_max * (z /(60/1.1)) ** alpha

    def reyolds_stress_tensor(self, z, u_0):
        """
        inputs in PU
        [I_x² * U_x_Mean², ~0, experiment]
        [~0, I_y² * U_y_Mean², ~0]
        [experiment , ~0, 9I_w² * U_z_Mean²]
        """
        house_length_pu = 10
        roof_height_pu = house_length_pu * 1.25
        stress = torch.ones(z.shape + (3, 3), device=self.units.lattice.device)
        z_0 = 0.02
        z_min = 1.2
        stress[..., 0, 0] = torch.where(z > z_min, ((1 - 2e-4 * (np.log10(z_0) + 3)**6)/torch.log(z/z_0))**2 * self.wind_speed_profile(z, u_0)**2, ((1-2e-4*(np.log10(z_0) + 3)**6)/np.log(z_min/z_0))**2 * self.wind_speed_profile(torch.tensor(z_min, device=self.units.lattice.device), u_0)**2)
        stress[..., 0, 1] = stress[..., 1, 0] = 0
        stress[..., 0, 2] = stress[..., 2, 0] = (0.4243 * (z/12.5 * 0.33) ** 2 - 2.288 * (z/12.5 * 0.33) - 2) * 1e-3 * u_0 **2 # function fittet to data from experiment
        stress[..., 1, 1] = (0.88/torch.log((z + 0.00001) * (0.33 / roof_height_pu) * 1e5 / 2.5))**2 * self.wind_speed_profile(z, u_0)**2
        stress[..., 1, 2] = stress[..., 2, 1] = 0
        stress[..., 2, 2] = 0.08**2 * self.wind_speed_profile(z, u_0)**2

        return stress * self.units.convert_density_to_pu(self.units.convert_pressure_pu_to_density_lu(0))

    @property
    def boundaries(self):
        x, y, z = self.grid
        L = self.L
        if self.N == 0:
            N = int(np.ceil((self.units.convert_length_to_pu(self.resolution_y) * self.units.convert_length_to_pu(self.resolution_z)) / (4 * L**2)))
        else:
            N = self.N
        print(f"N ist {N}")
        return [BounceBackBoundary(self.mask | (z < 1e-6), self.units.lattice),
                lt.ZeroGradientOutlet(self.units.lattice, [0, 0, 1]),
                lt.EquilibriumOutletP(self.units.lattice, [1, 0, 0], rho0=self.units.convert_pressure_pu_to_density_lu(0)),
                SyntheticEddyInlet(self.units.lattice, self.units, self.rgrid, rho=self.units.convert_density_to_pu(self.units.convert_pressure_pu_to_density_lu(0)), u_0=self.units.characteristic_velocity_pu, K=self.K_Factor*10, L=L, N=N, R=self.reyolds_stress_tensor, velocityProfile=self.wind_speed_profile)]


Re = int(sys.argv[1])
Ma = float(sys.argv[2])
identifier = sys.argv[3]
velocity_profile = sys.argv[4]
K_Factor = float(sys.argv[5])
N = int(sys.argv[6])
L = float(sys.argv[7])

print(f"Test of empty domain with {velocity_profile} velocity profile (K={K_Factor}, N={N}, L={L}) and new inlet, STILL NEED TO change stress[..., 0,2] to be with z/100")

house_length_lu = eg_x_length = 60
house_length_pu = 10 # METER
roof_height = house_length_lu*1.25
viscosity = 14.852989758837e-6 # bei 15°C und 1 bar
char_velocity = Re * viscosity / house_length_pu
print(char_velocity)
device = torch.device("cuda")
lattice = lt.Lattice(lt.D3Q27, device=device, dtype=torch.float32)
lattice.equilibrium = lt.QuadraticEquilibrium_LessMemory(lattice)
flow = Test3D(360, 240, 180, Re, Ma, lattice, char_length_lu=house_length_lu, char_length_pu=house_length_pu, char_density_pu=1.2250, char_velocity_pu=char_velocity, K_Factor=K_Factor, N=N, L=L, velocity_profile=velocity_profile)
collision = lt.KBCCollision3D(lattice, tau=flow.units.relaxation_parameter_lu)
streaming = lt.StandardStreaming(lattice)
simulation = lt.Simulation(flow, lattice, collision, streaming, nan_steps=61)
simulation.initialize_f_neq()
vtkrep = lt.VTKReporter(lattice, flow, interval=1000, start=0, filename_base=f"/scratch/mkliem3s/Boundary_{identifier}")
simulation.reporters.append(vtkrep)

u_x_inlet = [Statistics() for _ in range(0, flow.resolution_z)]
u_y_inlet = [Statistics() for _ in range(0, flow.resolution_z)]
u_z_inlet = [Statistics() for _ in range(0, flow.resolution_z)]

u_x_house = [Statistics() for _ in range(0, flow.resolution_z)]
u_y_house = [Statistics() for _ in range(0, flow.resolution_z)]
u_z_house = [Statistics() for _ in range(0, flow.resolution_z)]

time_start = time.time()
time_20k = time.time()

u_save = np.zeros([3, 3, flow.resolution_z, 300000])


for i in range(0,300000):
    simulation.step(1)
    if torch.any(torch.isnan(simulation.f)):
        print("Nan abort!!!")
        break
    u = flow.units.convert_velocity_to_pu(lattice.u(simulation.f)).cpu().numpy()
    u_save[:, 0, :, i] = u[:, 0, int(np.floor(flow.resolution_y / 2)), :]
    u_save[:, 1, :, i] = u[:, 90, int(np.floor(flow.resolution_y / 2)), :]
    u_save[:, 2, :, i] = u[:, 240, int(np.floor(flow.resolution_y / 2)), :]
    for j in range(0, flow.resolution_z):
        u_x_inlet[j].push(u[0, 0, int(np.floor(flow.resolution_y / 2)), j])
        u_y_inlet[j].push(u[1, 0, int(np.floor(flow.resolution_y / 2)), j])
        u_z_inlet[j].push(u[2, 0, int(np.floor(flow.resolution_y / 2)), j])
        # insert correct house x position!!
        u_x_house[j].push(u[0, 90, int(np.floor(flow.resolution_y / 2)), j])
        u_y_house[j].push(u[1, 90, int(np.floor(flow.resolution_y / 2)), j])
        u_z_house[j].push(u[2, 90, int(np.floor(flow.resolution_y / 2)), j])
    #save values every so often to avoid loss if aborted
    if i%20000 == 0:
        print(i)
        print("These 20k steps took {} minutes.".format((time_20k - time.time()) / 60))
        time_20k = time.time()
        u_house = np.zeros([3, 2, flow.resolution_z])
        u_inlet = np.zeros([3, 2, flow.resolution_z])

        for k in range(0, flow.resolution_z):
            u_house[0, 0, k] = u_x_house[k].mean()
            u_house[1, 0, k] = u_y_house[k].mean()
            u_house[2, 0, k] = u_z_house[k].mean()

            u_house[0, 1, k] = u_x_house[k].stddev()
            u_house[1, 1, k] = u_y_house[k].stddev()
            u_house[2, 1, k] = u_z_house[k].stddev()

            u_inlet[0, 0, k] = u_x_inlet[k].mean()
            u_inlet[1, 0, k] = u_y_inlet[k].mean()
            u_inlet[2, 0, k] = u_z_inlet[k].mean()

            u_inlet[0, 1, k] = u_x_inlet[k].stddev()
            u_inlet[1, 1, k] = u_y_inlet[k].stddev()
            u_inlet[2, 1, k] = u_z_inlet[k].stddev()

        np.save(f"u_house_{identifier}.npy", u_house)
        np.save(f"u_inlet_{identifier}.npy", u_inlet)
        np.save(f"u_save_{identifier}.npy", u_save)

time_end = time.time()
print("The whole simulation took {} hours.".format((time_end - time_start) / 3600))

#simulation.save_checkpoint(f"/scratch/mkliem3s/saves/BoundaryTest_save")
