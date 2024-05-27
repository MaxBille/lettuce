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

    def __init__(self, resolution_x, resolution_y, resolution_z, reynolds_number, mach_number, lattice, char_length_lu, char_length_pu, char_velocity_pu, char_density_pu, area, rank=0, size=1):
        self.rank = rank
        self.size = size
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.resolution_z = resolution_z
        self.mpiObject = lattice.mpiObject
        self.area = area
        self.units = lt.UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=char_length_lu, characteristic_length_pu=char_length_pu,
            characteristic_velocity_pu=char_velocity_pu,
            characteristic_density_pu=char_density_pu, characteristic_density_lu=1
        )
        self._mask = np.zeros(shape=(self.resolution_x, self.resolution_y, self.resolution_z), dtype=np.bool)
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
        self._mask = m.astype(np.bool)

    def initial_solution(self, x):
        p = np.zeros_like(x[0], dtype=float)[None, ...]
        #u_char = np.array([self.units.characteristic_velocity_pu, 0.0, 0.0])[..., None, None, None]
        #u = (1 - self.grid.select(self.mask.astype(np.float), self.rank)) * u_char
        u = np.zeros((len(x),) + x[0].shape)
        u[0] = self.wind_speed_profile(lattice.convert_to_tensor(np.where(self.mask, 0, x[2])), self.units.characteristic_velocity_pu).cpu().numpy()
        return p, u

    @property
    def boundaries(self): # inlet currently added via switch case based on call parameters!!!
        x, y, z = self.rgrid.global_grid()
        return [
            lt.BounceBackBoundary(self.mask | (z < 1e-6), self.units.lattice),
            lt.ZeroGradientOutlet(self.units.lattice, [0, 0, 1]),
            lt.EquilibriumOutletP(self.units.lattice, [1, 0, 0]),
            lt.SyntheticEddyInlet(self.units.lattice, self.units, self.rgrid, rho=self.units.convert_density_to_pu(self.units.convert_pressure_pu_to_density_lu(0)), u_0=self.units.characteristic_velocity_pu, K=10, L=0.75, N=150, R=self.reyolds_stress_tensor, velocityProfile=self.wind_speed_profile)]

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

    def wind_speed_profile_ref(self, z, u_max, alpha=0.25):
        # based on wikipedia article about wind profile
        # return u_max * (z / z_max) ** (1/alpha)
        # based on DIN Onstwaswindprofil
        # all inputs in PU
        # return 0.77 * u_max * (z / 10) ** (alpha)
        return u_max * (z / self.eg_height) ** alpha

    def wind_speed_profile(self, z, u_0):
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
        z_min = 1.2 # TODO Gleichung checken und Werte darin! # TODO unten windspeed profile  rausnehmen?
        stress[..., 0, 0] = torch.where(z > z_min, ((1 - 2e-4 * (np.log10(z_0) + 3)**6)/torch.log(z/z_0))**2 * self.wind_speed_profile(z, u_0)**2, ((1-2e-4*(np.log10(z_0) + 3)**6)/torch.log(torch.tensor(z_min/z_0, device=self.units.lattice.device)))**2 * self.wind_speed_profile(torch.tensor(z_min, device=self.units.lattice.device), u_0)**2)
        stress[..., 0, 1] = stress[..., 1, 0] = 0
        stress[..., 0, 2] = stress[..., 2, 0] = (0.4243 * (z/12.5 * 0.33) ** 2 - 2.288 * (z/12.5 * 0.33) - 2) * 1e-3 * u_0 **2 # function fittet to data from experiment
        stress[..., 1, 1] = (0.88/torch.log((z + 0.00001) * (0.33 / roof_height_pu) * 1e5 / 2.5))**2 * self.wind_speed_profile(z, u_0)**2 #**2 # TODO bei denen ist das nicht 0, aber meine stream wise mean velocity ist eifg schon 0
        stress[..., 1, 2] = stress[..., 2, 1] = 0
        stress[..., 2, 2] = 0.08**2 * self.wind_speed_profile(z, u_0)**2 # TODO bei denen ist das nicht 0, aber meine stream wise mean velocity ist eifg schon 0
        return stress

Re = int(sys.argv[1])
angle = float(sys.argv[2])
Ma = float(sys.argv[3])
identifier = sys.argv[4]
if len(sys.argv) >= 6:
    print(f"Changed Kf to {float(sys.argv[5])}")

house_length_lu = 60
house_length = 10

print("Starting at: ", datetime.now())
device = torch.device("cuda")
lattice = lt.Lattice(lt.D3Q27, device=device, dtype=torch.float32)
lattice.equilibrium = lt.QuadraticEquilibrium_LessMemory(lattice)
viscosity = 14.852989758837 * 10**(-6) # bei 15°C und 1 bar
char_velocity = Re * viscosity / house_length
flow = HouseFlow3D(360, 240, 180, Re, Ma, lattice, char_length_lu=house_length_lu, char_length_pu=house_length, char_density_pu=1.2250, char_velocity_pu=char_velocity, area=house_length_lu**2)
#flow.mask, points = flow.house([flow.units.convert_length_to_pu(int(a * b)) for a, b in zip([1/3, 0.5, 0], flow.grid.shape)], house_length, house_length, house_length/1.1, house_length, house_length, angle=angle)
flow.mask, points = flow.house3([120,  120, 0], house_length_lu, house_length_lu, 60/1.1, 0, angle=angle)
collision = lt.KBCCollision3D(lattice, tau=flow.units.relaxation_parameter_lu)
streaming = lt.StandardStreaming(lattice)
simulation = lt.Simulation(flow, lattice, collision, streaming, nan_steps=61)
simulation.initialize_f_neq()
#vtkrep = lt.VTKReporter(lattice, flow, interval=500, filename_base=f"/scratch/mkliem3s/verify/Haus_verify_{identifier}_{angle}deg_Re{Re}")
#simulation.reporters.append(vtkrep)
#vtkrepP = lt.VTKReporterP(lattice, flow, interval=125, filename_base=f"/scratch/mkliem3s/verify/Haus_verify_{identifier}_{angle}deg_Re{Re}")
#simulation.reporters.append(vtkrepP)
#vtkrep.output_mask(simulation.no_collision_mask)

press = lt.LocalPressure(lattice, flow, points)
observ = lt.ObservableReporter(press, 1, None)
simulation.reporters.append(observ)

print(f"Parameters: Re: {Re}, Ma {Ma}, House_length_LU: {house_length_lu} Identifier: {identifier}")
print(f"Relaxation parameter: {simulation.flow.units.relaxation_parameter_lu}")
print(f"Duration of a time step in PU: {simulation.flow.units.convert_time_to_pu(1)}")
print(f"Characteristic velocity: {char_velocity}")

time_start = time.time()
for i in range(0, 10):
    print('MLUPS: ', simulation.step(2 * 10 ** 4))
    print(i*2e4)
    np.save(f"/scratch/mkliem3s/verify/p_verify_{identifier}_{angle}deg_Re{Re}.npy", observ.out)
time_end = time.time()
print("The simulation took {} hours.".format((time_end - time_start) / 3600))

simulation.save_checkpoint(f"/scratch/mkliem3s/saves/Haus_verify_{identifier}_{angle}deg_Re{Re}_save")


