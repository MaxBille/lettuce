import lettuce as lt
import numpy as np
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from datetime import datetime
import sys
import time

class HouseFlow3D(object):
    """
    Flow class to simulate the flow around an object (mask) in 3D.
    See documentation for :class:`~Obstacle2D` for details.
    """

    def __init__(self, resolution_x, resolution_y, resolution_z, reynolds_number, mach_number, lattice, char_length_lu, char_length_pu, char_velocity_pu, char_density_pu, area, K_Factor, L, N, velocity_profile):
        self.rank = 0
        self.size = 1
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
        self.area = area
        self.units = lt.UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=char_length_lu, characteristic_length_pu=char_length_pu,
            characteristic_velocity_pu=char_velocity_pu,
            characteristic_density_pu=char_density_pu, characteristic_density_lu=1
        )
        self._mask = np.zeros(shape=(self.resolution_x, self.resolution_y, self.resolution_z), dtype=bool)
        self.rgrid = lt.RegularGrid([resolution_x, resolution_y, resolution_z], self.units.characteristic_length_lu,
                                self.units.characteristic_length_pu, endpoint=False)

    @property
    def mask(self):
        return self._mask

    @property
    def grid(self):
        return self.rgrid()

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, m):
        assert isinstance(m, np.ndarray) and m.shape == self.rgrid.global_shape
        self._mask = m.astype(bool)

    def initial_solution(self, x):
        p = np.zeros_like(x[0], dtype=float)[None, ...]
        #u_char = np.array([self.units.characteristic_velocity_pu, 0.0, 0.0])[..., None, None, None]
        #u = (1 - self.grid.select(self.mask.astype(np.float), self.rank)) * u_char
        u = np.zeros((len(x),) + x[0].shape)
        u[0] = self.wind_speed_profile(lattice.convert_to_tensor(np.where(self.mask, 0, x[2])), self.units.characteristic_velocity_pu).cpu().numpy()
        return p, u

    @property
    def boundaries(self):
        x, y, z = self.grid
        L = self.L
        if self.N == 0:
            N = int(np.ceil((self.units.convert_length_to_pu(self.resolution_y) * self.units.convert_length_to_pu(self.resolution_z)) / (4 * L**2)))
        else:
            N = self.N
        print(f"N ist {N}")
        return [lt.BounceBackBoundary(self.mask | (z < 1e-6), self.units.lattice),
                lt.ZeroGradientOutlet(self.units.lattice, [0, 0, 1]),
                lt.EquilibriumOutletP(self.units.lattice, [1, 0, 0], rho0=self.units.convert_pressure_pu_to_density_lu(0)),
                lt.SyntheticEddyInlet(self.units.lattice, self.units, self.rgrid, rho=self.units.convert_density_to_pu(self.units.convert_pressure_pu_to_density_lu(0)), u_0=self.units.characteristic_velocity_pu, K=self.K_Factor*10, L=L, N=N, R=self.reyolds_stress_tensor, velocityProfile=self.wind_speed_profile)
		]

    def house3(self, o, eg_x_length, eg_y_length, eg_height, roof_overhang, roof_width=0, angle=45):
        # In LU, weil da sonst immer alles nicht stimmt^^ alles außer Winkel ist jetzt in LU roof_width gibts nich
        angle = angle * np.pi / 180
        self.eg_height = int(eg_height)
        eg_height = self.eg_height
        roof_width = eg_y_length
        inside_mask = np.zeros_like(self.rgrid.global_grid()[0], dtype=bool)
        inside_mask[int(o[0] - eg_x_length / 2):int(o[0] + eg_x_length / 2),
        int(o[1] - eg_y_length / 2):int(o[1] + eg_y_length / 2), :eg_height] = True
        inside_mask[int(o[0] - eg_x_length / 2 - roof_overhang):int(o[0] + eg_x_length / 2 + roof_overhang),
        int(o[1] - eg_y_length / 2):int(o[1] + eg_y_length / 2), eg_height] = True
        inside_mask[int(o[0] - eg_x_length / 2 - roof_overhang):int(o[0] + eg_x_length / 2 + roof_overhang),
        int(o[1] - eg_y_length / 2):int(o[1] + eg_y_length / 2), eg_height + 1:] = \
            np.where(np.logical_and(self.units.convert_length_to_lu(
                self.grid[2][int(o[0] - eg_x_length / 2 - roof_overhang):int(o[0] + eg_x_length / 2 + roof_overhang),
                int(o[1] - eg_y_length / 2):int(o[1] + eg_y_length / 2),
                eg_height + 1:]) < o[2] + eg_height + 0.001 - np.tan(angle) * (self.units.convert_length_to_lu(
                self.grid[0][int(o[0] - eg_x_length / 2 - roof_overhang):int(o[0] + eg_x_length / 2 + roof_overhang),
                int(o[1] - eg_y_length / 2):int(o[1] + eg_y_length / 2),
                eg_height + 1:]) - o[0] - eg_x_length / 2), self.units.convert_length_to_lu(
                self.grid[2][int(o[0] - eg_x_length / 2 - roof_overhang):int(
                    o[0] + eg_x_length / 2 + roof_overhang),
                int(o[1] - eg_y_length / 2):int(o[1] + eg_y_length / 2),
                eg_height + 1:])< o[2] + eg_height + 0.001 + np.tan(angle) * (self.units.convert_length_to_lu(
                self.grid[0][int(o[0] - eg_x_length / 2 - roof_overhang):int(
                    o[0] + eg_x_length / 2 + roof_overhang),
                int(o[1] - eg_y_length / 2):int(o[1] + eg_y_length / 2),
                eg_height + 1:]) - o[0] + eg_x_length / 2)), True, inside_mask[int(
                o[0] - eg_x_length / 2 - roof_overhang):int(o[0] + eg_x_length / 2 + roof_overhang), int(
                o[1] - eg_y_length / 2):int(o[1] + eg_y_length / 2), eg_height + 1:])

        measurement_points = []
        for x in range(0, eg_x_length+1):
            _x = o[0] - int(eg_x_length / 2) + x
            measurement_points.append([_x, o[1], eg_height + np.tan(angle) * (eg_x_length/2 - np.abs(o[0] - _x)) + 3])

        return inside_mask, measurement_points

    def wind_speed_profile_turb(self, z, u_0):
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
        return u_max * (z / self.units.convert_length_to_pu(self.eg_height)) ** alpha

    def reyolds_stress_tensor(self, z, u_0):
        """
        inputs in PU
        [I_x² * U_x_Mean², ~0, experiment]
        [~0, I_y² * U_y_Mean², ~0]
        [experiment , ~0, 9I_w² * U_z_Mean²]
        """
        house_length_pu = 10
        roof_height_pu = self.units.convert_length_to_pu(self.eg_height) #house_length_pu * 1.25
        stress = torch.ones(z.shape + (3, 3), device=self.units.lattice.device)
        z_0 = 0.02
        z_min = 1.2
        stress[..., 0, 0] = torch.where(z > z_min, ((1 - 2e-4 * (np.log10(z_0) + 3)**6)/torch.log(z/z_0))**2 * self.wind_speed_profile(z, u_0)**2, ((1-2e-4*(np.log10(z_0) + 3)**6)/np.log(z_min/z_0))**2 * self.wind_speed_profile(torch.tensor(z_min, device=self.units.lattice.device), u_0)**2)
        stress[..., 0, 1] = stress[..., 1, 0] = 0
        stress[..., 0, 2] = stress[..., 2, 0] = (0.4243 * (z/100) ** 2 - 2.288 * (z/100) - 2) * 1e-3 * u_0 **2 # function fittet to data from experiment
        stress[..., 1, 1] = (0.88/torch.log((z + 0.00001) * (0.33 / roof_height_pu) * 1e5 / 2.5))**2 * self.wind_speed_profile(z, u_0)**2
        stress[..., 1, 2] = stress[..., 2, 1] = 0
        stress[..., 2, 2] = 0.08**2 * self.wind_speed_profile(z, u_0)**2

        return stress * self.units.convert_density_to_pu(self.units.convert_pressure_pu_to_density_lu(0))

Re = int(sys.argv[1])
Ma = float(sys.argv[2])
identifier = sys.argv[3]
velocity_profile = sys.argv[4]
K_Factor = float(sys.argv[5])
N = int(sys.argv[6])
L = float(sys.argv[7])
angle = float(sys.argv[8])

house_length_lu = 60
house_length_pu = 10

print(f"References house with {velocity_profile} velocity profile (angle={angle}, K={K_Factor}, N={N}, L={L}) and new inlet, stress[..., 0,2] with z/100")
print("Starting at: ", datetime.now())
viscosity = 14.852989758837e-6 # bei 15°C und 1 bar
char_velocity = Re * viscosity / house_length_pu
print(char_velocity)
lattice = lt.Lattice(lt.D3Q27, device=torch.device("cpu"), dtype=torch.float32)
lattice.equilibrium = lt.QuadraticEquilibrium_LessMemory(lattice)
lattice_gpu = lt.Lattice(lt.D3Q27, device=torch.device("cuda"), dtype=torch.float32)
flow = HouseFlow3D(360, 240, 180, Re, Ma, lattice, char_length_lu=house_length_lu, char_length_pu=house_length_pu, char_density_pu=1.2250, char_velocity_pu=char_velocity, area=house_length_lu**2, K_Factor=K_Factor, N=N, L=L, velocity_profile=velocity_profile)
flow.mask, points = flow.house3([120,  120, 0], house_length_lu, house_length_lu, house_length_lu/1.1, 0, angle=angle)
collision = lt.KBCCollision3D(lattice_gpu, tau=flow.units.relaxation_parameter_lu)
streaming = lt.StandardStreaming(lattice)
simulation = lt.Simulation(flow, lattice, collision, streaming, nan_steps=61)
simulation.initialize_f_neq()
vtkrep = lt.VTKReporter(lattice, flow, interval=200, filename_base=f"/media/martin/Scratch/TestLokal/Haus_verify_{identifier}_{angle}deg_Re{Re}")
simulation.reporters.append(vtkrep)
#vtkrepP = lt.VTKReporterP(lattice, flow, interval=125, filename_base=f"/scratch/mkliem3s/verify/Haus_verify_{identifier}_{angle}deg_Re{Re}")
#simulation.reporters.append(vtkrepP)
vtkrep.output_mask(simulation.no_collision_mask)

press = lt.LocalPressure(lattice, flow, points)
observ = lt.ObservableReporter(press, 1, None)
simulation.reporters.append(observ)

u_points = [[0, 120, z] for z in range(0, 180)]
u_points += [[70, 120, z] for z in range(0, 180)]
u = lt.LocalVelocity(lattice, flow, u_points)
observ2 = lt.ObservableReporter(u, 1, None)
simulation.reporters.append(observ2)

print(f"Parameters: Re: {Re}, Ma {Ma}, House_length_LU: {house_length_lu} Identifier: {identifier}")
print(f"Relaxation parameter: {simulation.flow.units.relaxation_parameter_lu}")
print(f"Duration of a time step in PU: {simulation.flow.units.convert_time_to_pu(1)}")
print(f"Characteristic velocity: {char_velocity}")

time_start = time.time()
for i in range(0, 15):
    print('MLUPS: ', simulation.step(2 * 10 ** 4))
    print((i + 1)*2e4)
    np.save(f"/scratch/mkliem3s/verify/u_verify_{identifier}_{angle}deg.npy", np.array(observ2.out)[:, 2:].reshape([(i + 1)*2e4, 3, 2, 180]))
    np.save(f"/scratch/mkliem3s/verify/p_verify_{identifier}_{angle}deg.npy", observ.out)
time_end = time.time()
print("The simulation took {} hours.".format((time_end - time_start) / 3600))

simulation.save_checkpoint(f"/scratch/mkliem3s/saves/Haus_verify_{identifier}_{angle}deg_Re{Re}_save")

