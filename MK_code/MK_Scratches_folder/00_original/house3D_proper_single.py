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
        self.area = area
        self.units = lt.UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=char_length_lu, characteristic_length_pu=char_length_pu,
            characteristic_velocity_pu=char_velocity_pu,
            characteristic_density_pu=char_density_pu, characteristic_density_lu=1
        )
        self._mask = np.zeros(shape=(self.resolution_x, self.resolution_y, self.resolution_z), dtype=np.bool)
        self.grid = lt.RegularGrid([resolution_x, resolution_y, resolution_z], self.units.characteristic_length_lu,
                                self.units.characteristic_length_pu, endpoint=False, rank=rank, size=size)

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, m):
        assert isinstance(m, np.ndarray) and m.shape == self.grid.global_shape
        self._mask = m.astype(np.bool)

    def initial_solution(self, x):
        p = np.zeros_like(x[0], dtype=float)[None, ...]
        #u_char = np.array([self.units.characteristic_velocity_pu, 0.0, 0.0])[..., None, None, None]
        #u = (1 - self.grid.select(self.mask.astype(np.float), self.rank)) * u_char
        u = np.zeros((len(x),) + x[0].shape)
        u[0] = self.wind_speed_profile(np.where(self.mask, 0, x[2]), self.units.characteristic_velocity_pu, 2)
        return p, u

    @property
    def boundaries(self): # inlet currently added via switch case based on call parameters!!!
        x, y, z = self.grid.global_grid()
        p, u = self.initial_solution(self.grid())
        b = [
            lt.BounceBackBoundary(self.mask | (z < 1e-6), self.units.lattice),
            lt.ZeroGradientOutlet(self.units.lattice, [0, 0, 1]),
            lt.EquilibriumOutletP(self.units.lattice, [1, 0, 0]),
            # | ((z < 1e-6) & (x < np.max(x).item()*0.9))
            #lt.EquilibriumBoundaryPU(
            #    np.abs(x) < 1e-6, self.units.lattice, self.units, u[:, 0, ...], p[0, 0, ...]
            #),
            #lt.EquilibriumInletU(self.units.lattice, [-1, 0, 0], self.units, u[:, 0,...]),
            #lt.NonEquilibriumExtrapolationOutlet(self.units.lattice, 1, [0, 1, 0]),
            #lt.ZeroGradientOutlet(self.units.lattice, [0, 1, 0]),
            #lt.ZeroGradientOutlet(self.units.lattice, [0, -1, 0]),
            #lt.EquilibriumOutletP(self.units.lattice, [0, 0, 1]),
            #lt.EquilibriumOutlet(self.units.lattice, [0, -1, 0]),
            #lt.BounceBackVelocityInlet(self.units.lattice, self.units, [-1, 0, 0], u[:, 0, ...]),
            #lt.NonEquilibriumExtrapolationInletU(self.units.lattice, self.units, [-1, 0, 0], np.array(u)),
            #lt.HalfWayBounceBackWall([0, 0, -1], lattice),
        ]
        if "ggwPU" in identifier:
            b.append(lt.EquilibriumBoundaryPU(np.abs(x) < 1e-6, self.units.lattice, self.units, u[:, 0, ...], p[0, 0, ...]))
        elif "ggwU" in identifier:
            b.append(lt.EquilibriumInletU(self.units.lattice, [-1, 0, 0], self.units, u[:, 0,...]))
        elif "bbI" in identifier:
            b.append(lt.BounceBackVelocityInlet(self.units.lattice, self.units, [-1, 0, 0], u[:, 0, ...]))
        else:
            b.append(lt.NonEquilibriumExtrapolationInletU(self.units.lattice, self.units, [-1, 0, 0], np.array(u)))
        return b

    def house(self, o, eg_length, eg_width, eg_height, roof_length, roof_width, angle):
        o[2] = 0
        angle = angle * np.pi / 180
        inside_mask = np.zeros_like(self.grid.global_grid()[0], dtype=bool)
        inside_mask = np.where(np.logical_and(
            np.logical_and(np.logical_and(o[0] - eg_length / 2 < self.grid.global_grid()[0], self.grid.global_grid()[0] < o[0] + eg_length / 2),
                           np.logical_and(o[1] - eg_width / 2 < self.grid.global_grid()[1], self.grid.global_grid()[1] < o[1] + eg_width / 2)),
                            np.logical_and(o[2] - eg_height <= self.grid.global_grid()[2], self.grid.global_grid()[2] <= o[2] + eg_height)), True, inside_mask)
        inside_mask = np.where(np.logical_and(np.logical_and(
                        np.logical_and(np.logical_and(o[0] - roof_length / 2 < self.grid.global_grid()[0], self.grid.global_grid()[0] < o[0] + roof_length / 2),
                        np.logical_and(o[1] - roof_width / 2 < self.grid.global_grid()[1], self.grid.global_grid()[1] < o[1] + roof_width / 2)),
                        np.logical_and(o[2] + eg_height < self.grid.global_grid()[2],
                        self.grid.global_grid()[2] < o[2] + eg_height + 0.001 + np.tan(angle) * (self.grid.global_grid()[0] - o[0] + roof_width / 2))),
                        self.grid.global_grid()[2] < o[2] + eg_height + 0.001 - np.tan(angle) * (self.grid.global_grid()[0] - o[0] - roof_width / 2)), True, inside_mask)

        """make masks for fs to be bounced / not streamed by going over all obstacle points and
        following all e_i's to find neighboring points and which of their fs point towards the obstacle
        (fs pointing to obstacle are added to no_stream_mask, fs pointing away are added to bouncedFs)"""

        x, y, z = inside_mask.shape
        outgoing_mask = np.zeros((self.units.lattice.Q, x, y, z), dtype=bool)
        a, b, c = np.where(inside_mask)
        for p in range(0, len(a)):
            for i in range(0, self.units.lattice.Q):
                try:  # try in case the neighboring cell does not exist (an f pointing out of simulation domain)
                    if not inside_mask[a[p] + self.units.lattice.stencil.e[i, 0], b[p] + self.units.lattice.stencil.e[i, 1], c[p] + self.units.lattice.stencil.e[i, 2]]:
                        outgoing_mask[i, a[p] + self.units.lattice.stencil.e[i, 0], b[p] + self.units.lattice.stencil.e[i, 1], c[p] + self.units.lattice.stencil.e[i, 2]] = 1
                except IndexError:
                    pass  # just ignore this iteration since there is no neighbor there

        measurement_points = []
        measurement_points.append(o + np.asarray([roof_length * 0.25, -roof_width * 0.25, 0]))
        measurement_points.append(o + np.asarray([roof_length * 0.75, -roof_width * 0.25, 0]))
        measurement_points.append(o + np.asarray([roof_length * 0.25, roof_width * 0.25, 0]))
        measurement_points.append(o + np.asarray([roof_length * 0.75, roof_width * 0.25, 0]))
        measurement_points.append(o + np.asarray([roof_length * 0.5, 0, 0]))

        for point in measurement_points:
            point[2] = eg_height + 0.001 - np.tan(angle) * (point[0] - o[0] - roof_width / 2)
            for i in range(0, 3):
                point[i] = int(self.units.convert_length_to_lu(point[i]))
            point[2] += 2

        return inside_mask, measurement_points

    def house2(self, o, eg_x_length, eg_y_length, roof_height, roof_overhang, roof_width=0, angle=45):
        if roof_width == 0:
            roof_width=eg_y_length
        o[2] = 0
        angle = angle * np.pi / 180
        eg_height = roof_height - (eg_x_length / 2 + roof_overhang) * np.tan(angle)
        self.eg_height = eg_height
        inside_mask = np.zeros_like(self.grid.global_grid()[0], dtype=bool)
        inside_mask = np.where(np.logical_and(
            np.logical_and(np.logical_and(o[0] - eg_x_length / 2 < self.grid.global_grid()[0], self.grid.global_grid()[0] < o[0] + eg_x_length / 2),
                           np.logical_and(o[1] - eg_y_length / 2 < self.grid.global_grid()[1], self.grid.global_grid()[1] < o[1] + eg_y_length / 2)),
                            np.logical_and(o[2] - eg_height <= self.grid.global_grid()[2], self.grid.global_grid()[2] <= o[2] + eg_height)), True, inside_mask)
        inside_mask = np.where(np.logical_and(
                        np.logical_and(np.logical_and(o[0] - eg_x_length / 2 - roof_overhang < self.grid.global_grid()[0], self.grid.global_grid()[0] < o[0] + eg_x_length / 2 + roof_overhang),
                        np.logical_and(o[1] - roof_width / 2 < self.grid.global_grid()[1], self.grid.global_grid()[1] < o[1] + roof_width / 2)),
                        np.logical_and(o[2] + eg_height < self.grid.global_grid()[2],
                        self.grid.global_grid()[2] < o[2] + roof_height + 0.001 - np.tan(angle) * np.abs(self.grid.global_grid()[0] - o[0]))), True, inside_mask)
        #inside_mask = np.where(np.logical_and(
        #                np.logical_and(np.logical_and(o[0] - eg_x_length / 2 - roof_overhang < self.grid.global_grid()[0],self.grid.global_grid()[0] < o[0] + eg_x_length / 2 + roof_overhang),
        #                np.logical_and(o[1] - roof_width / 2 < self.grid.global_grid()[1],self.grid.global_grid()[1] < o[1] + roof_width / 2)),
        #                np.logical_and(o[2] + eg_height < self.grid.global_grid()[2],
        #                self.grid.global_grid()[2] < o[2] + eg_height + self.units.convert_length_to_pu(1.5))), True, inside_mask)

        """make masks for fs to be bounced / not streamed by going over all obstacle points and
        following all e_i's to find neighboring points and which of their fs point towards the obstacle
        (fs pointing to obstacle are added to no_stream_mask, fs pointing away are added to bouncedFs)"""

        x, y, z = inside_mask.shape
        outgoing_mask = np.zeros((self.units.lattice.Q, x, y, z), dtype=bool)
        a, b, c = np.where(inside_mask)
        for p in range(0, len(a)):
            for i in range(0, self.units.lattice.Q):
                try:  # try in case the neighboring cell does not exist (an f pointing out of simulation domain)
                    if not inside_mask[a[p] + self.units.lattice.stencil.e[i, 0], b[p] + self.units.lattice.stencil.e[i, 1], c[p] + self.units.lattice.stencil.e[i, 2]]:
                        outgoing_mask[i, a[p] + self.units.lattice.stencil.e[i, 0], b[p] + self.units.lattice.stencil.e[i, 1], c[p] + self.units.lattice.stencil.e[i, 2]] = 1
                except IndexError:
                    pass  # just ignore this iteration since there is no neighbor there

        #auf dem Dach:
        measurement_points = []
        measurement_points.append(o + np.asarray([(eg_y_length / 2 + roof_overhang) * 0.25, -roof_width * 0.25, 0]))
        measurement_points.append(o + np.asarray([(eg_y_length / 2 + roof_overhang) * 0.75, -roof_width * 0.25, 0]))
        measurement_points.append(o + np.asarray([(eg_y_length / 2 + roof_overhang) * 0.25, roof_width * 0.25, 0]))
        measurement_points.append(o + np.asarray([(eg_y_length / 2 + roof_overhang) * 0.75, roof_width * 0.25, 0]))
        measurement_points.append(o + np.asarray([(eg_y_length / 2 + roof_overhang) * 0.5, 0, 0]))

        #auf der Vorderseite vom Dach?

        measurement_points.append(o - np.asarray([(eg_y_length / 2 + roof_overhang) * 0.25, -roof_width * 0.25, 0]))
        measurement_points.append(o - np.asarray([(eg_y_length / 2 + roof_overhang) * 0.75, -roof_width * 0.25, 0]))
        measurement_points.append(o - np.asarray([(eg_y_length / 2 + roof_overhang) * 0.25, roof_width * 0.25, 0]))
        measurement_points.append(o - np.asarray([(eg_y_length / 2 + roof_overhang) * 0.75, roof_width * 0.25, 0]))
        measurement_points.append(o - np.asarray([(eg_y_length / 2 + roof_overhang) * 0.5, 0, 0]))

        for point in measurement_points:
            point[2] = o[2] + roof_height + 0.001 - np.tan(angle) * np.abs(point[0] - o[0])
            for i in range(0, 3):
                point[i] = int(self.units.convert_length_to_lu(point[i]))
            point[2] += 2

        measurement_points += [point + np.array([0, 0, 8]) for point in measurement_points[0:10]]
        measurement_points += [point + np.array([0, 0, 18]) for point in measurement_points[0:10]]

            # im freien Raum vor dem Haus

        measurement_points.append(np.asarray([int(self.units.convert_length_to_lu(x)) for x in [self.grid.shape_pu[0] * 0.05, self.grid.shape_pu[1] / 2, self.grid.shape_pu[2] / 2]]))

        return inside_mask, measurement_points

    def house3(self, o, eg_x_length, eg_y_length, eg_height, roof_overhang, roof_width=0, angle=45):
        # In LU, weil da sonst immer alles nicht stimmt^^ alles außer Winkel ist jetzt in LU roof_width gibts nich
        angle = angle * np.pi / 180
        self.eg_height = eg_height
        roof_width = eg_y_length
        inside_mask = np.zeros_like(self.grid.global_grid()[0], dtype=bool)
        inside_mask[int(o[0] - eg_x_length / 2):int(o[0] + eg_x_length / 2),
        int(o[1] - eg_y_length / 2):int(o[1] + eg_y_length / 2), :eg_height] = True
        inside_mask[int(o[0] - eg_x_length / 2 - roof_overhang):int(o[0] + eg_x_length / 2 + roof_overhang),
        int(o[1] - eg_y_length / 2):int(o[1] + eg_y_length / 2), eg_height] = True
        inside_mask[int(o[0] - eg_x_length / 2 - roof_overhang):int(o[0] + eg_x_length / 2 + roof_overhang),
        int(o[1] - eg_y_length / 2):int(o[1] + eg_y_length / 2), eg_height + 1:] = \
            np.where(np.logical_and(self.units.convert_length_to_lu(
                self.grid()[2][int(o[0] - eg_x_length / 2 - roof_overhang):int(o[0] + eg_x_length / 2 + roof_overhang),
                int(o[1] - eg_y_length / 2):int(o[1] + eg_y_length / 2),
                eg_height + 1:]) < o[2] + eg_height + 0.001 - np.tan(angle) * (self.units.convert_length_to_lu(
                self.grid()[0][int(o[0] - eg_x_length / 2 - roof_overhang):int(o[0] + eg_x_length / 2 + roof_overhang),
                int(o[1] - eg_y_length / 2):int(o[1] + eg_y_length / 2),
                eg_height + 1:]) - o[0] - eg_x_length/2),self.units.convert_length_to_lu(self.grid()[2][int(o[0] - eg_x_length / 2 - roof_overhang):int(
                                            o[0] + eg_x_length / 2 + roof_overhang),
                                        int(o[1] - eg_y_length / 2):int(o[1] + eg_y_length / 2),
                                        eg_height + 1:])
                                    < o[2] + eg_height + 0.001 + np.tan(angle) * (self.units.convert_length_to_lu(
                                        self.grid()[0][int(o[0] - eg_x_length / 2 - roof_overhang):int(
                                            o[0] + eg_x_length / 2 + roof_overhang),
                                        int(o[1] - eg_y_length / 2):int(o[1] + eg_y_length / 2),
                                        eg_height + 1:]) - o[0] + eg_x_length/2)), True, inside_mask[int(
                o[0] - eg_x_length / 2 - roof_overhang):int(o[0] + eg_x_length / 2 + roof_overhang), int(
                o[1] - eg_y_length / 2):int(o[1] + eg_y_length / 2), eg_height + 1:])
        x = np.linspace(-0.1, 0.5, 7, endpoint=True)
        y = np.linspace(-0.5, 0.5, 11, endpoint=True)
        xy = np.meshgrid(y, x)
        # xy[x/y, x, y]
        measurement_points = np.asarray([(eg_x_length / 2 + roof_overhang) * xy[1], eg_y_length * xy[0], np.zeros_like(xy[0])])

        meas_points = []
        meas_points2 = []
        meas_points3 = []
        meas_points4 = []

        if 0:
            for y in range(0, 10):
                for x in range(0, 6):
                    measurement_points[:, x, y] = measurement_points[:, x, y] + o
                    measurement_points[2, x, y] = roof_height + 0.001 - np.tan(angle) * np.abs(measurement_points[0, x, y] - o[0])
                    measurement_points[:, x, y] = measurement_points[:, x, y].astype(np.int32)
                    meas_points.append(measurement_points[:, x, y].astype(np.int32))
                    meas_points2.append(measurement_points[:, x, y] + np.asarray([0, 0, 1]))
                    meas_points3.append(measurement_points[:, x, y] + np.asarray([0, 0, 2]))
                    meas_points4.append(measurement_points[:, x, y] + np.asarray([0, 0, 3]))

            measurement_points = meas_points + meas_points2 + meas_points3  + meas_points4

            middle_front = np.asarray([(eg_x_length / 2 + roof_overhang) * -0.25, 0, roof_height + 0.001 - np.tan(angle) * np.abs((eg_x_length / 2 + roof_overhang) * -0.25)]) + np.asarray(o)
            for i in range(0, 3):
                measurement_points.append(middle_front.astype(np.int32) + np.asarray([0, 0, i]))
            # Hinter dem Einlass
            measurement_points.append(np.asarray([int(self.units.convert_length_to_lu(x)) for x in
                                                  [self.grid.shape_pu[0] * 0.05, self.grid.shape_pu[1] / 2,
                                                   self.grid.shape_pu[2] / 2]]))
        measurement_points=[]
        return inside_mask, measurement_points

    def wind_speed_profile(self, z, u_max, alpha=0.25):
        # based on wikipedia article about wind profile
        #return u_max * (z / z_max) ** (1/alpha)
        #based on DIN Onstwaswindprofil
        # all inputs in PU
        #return 0.77 * u_max * (z / 10) ** (alpha)
        return u_max * (z/self.eg_height) ** alpha

Re = 20000 #int(sys.argv[1])
angle = 50 #int(sys.argv[2])
identifier = "ggwPUtest" #sys.argv[3]
house_length_lu = 60
house_length = 10

"""
reynolds_number
mach_number = 0.05
relaxation parameter: wird automatisch passend zur Empfehlung berechnet (solange dx (LU) = 1 ist
characteristic_length_pu: Länge des Hauses in m, z.B: 12.5m?
characteristic_velocity_pu: Geschw. der Anströmung in m/s
characteristic_length_lu: Seitenlänge des Hauses in Gitterpunkten
origin_pu=None: ins Haus legen?
characteristic_density_lu=1 rho_0 = 1!!!
characteristic_density_pu=1 ist wichtig für die Density umrechnung... 1 atm? abh. von T und Feuchte... jeweils grobe Werte aussuchen.. 15°C und 80% Feuchte?
"""

print("Starting at: ", datetime.now())
device = torch.device("cpu")
lattice = lt.Lattice(lt.D3Q27, device=device, dtype=torch.float32)
lattice.equilibrium = lt.QuadraticEquilibrium_LessMemory(lattice)
viscosity = 14.852989758837 * 10**(-6) # bei 15°C und 1 bar
char_velocity = Re * viscosity / house_length
flow = HouseFlow3D(100, 100, 120, Re, 0.1, lattice, char_length_lu=house_length_lu, char_length_pu=house_length, char_density_pu=1.2250, char_velocity_pu=char_velocity, area=house_length_lu**2)
flow.mask, points = flow.house2([flow.units.convert_length_to_pu(int(a * b)) for a, b in zip([0.5, 0.5, 0], flow.grid.shape)], 1 * house_length, 1 * house_length, 1.5 * house_length, 0.15 * house_length, angle=angle)
test, pointz = flow.house([flow.units.convert_length_to_pu(int(a * b)) for a, b in zip([0.5, 0.5, 0], flow.grid.shape)], house_length, house_length, house_length, house_length, house_length, angle=angle)
collision = lt.KBCCollision3D(lattice, tau=flow.units.relaxation_parameter_lu)
streaming = lt.StandardStreaming(lattice)
simulation = lt.Simulation(flow, lattice, collision, streaming, nan_steps=61)
#simulation.initialize_f_neq()
vtkrep = lt.VTKReporter(lattice, flow, interval=1000, filename_base=f"/mnt/hgfs/VMshare/Haus_proper_{identifier}_{angle}deg_Re{Re}", nan_out=True)
simulation.reporters.append(vtkrep)

# Irgendwann mal die messpunkte als Mask-Point ausgeben
maske = simulation.no_collision_mask
#for point in points:
#    maske[point] = True
test = lattice.convert_to_tensor(test)
vtkrep.output_mask(test)

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import  pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for angle in range(0,60, 5):
     print("test")
     test_mask, test_points = flow.house3([50,  50, 0], 60, 60, 60, 0, angle=angle)
     test_mask = lattice.convert_to_tensor(test_mask)
     vtkrep.filename_base=f"/mnt/hgfs/VMshare/test_{angle}_"
     vtkrep.output_mask(test_mask)
     test_points = np.asarray(test_points)
     print(f"masked {angle}")
#ax.plot(test_points[:, 0], test_points[:, 1], test_points[:, 2], "bo")
#plt.show()
if 0:
    mass = lt.Mass(lattice, flow, flow.boundaries[0].make_no_collision_mask(flow.grid.shape))
    massrep = lt.ObservableReporter(mass, 1, None)
    simulation.reporters.append(massrep)

    press = lt.LocalPressure(lattice, flow, points) #[lt.LocalPressure(lattice, flow, points) for point in points]
    observ = lt.ObservableReporter(press, 1, None)#open(f"/mnt/hgfs/VMshare/p_proper_{identifier}_{angle}deg_Re{Re}.txt", "w")) #[lt.ObservableReporter(obs, 1, None) for obs in press]
    simulation.reporters.append(observ) #simulation.reporters + observ

    time_start = time.time()

    print(f"Relaxation parameter: {simulation.flow.units.relaxation_parameter_lu}")
    print(f"Duration of a time step in PU: {simulation.flow.units.convert_time_to_pu(1)}")
    print(f"Characteristic velocity: {char_velocity}")

    print('MLUPS: ', simulation.step(100))

    time_end = time.time()
    print("The simulation took {} hours.".format((time_end - time_start) / 3600))

    np.save(f"/mnt/hgfs/VMshare/p_proper_{identifier}_{angle}deg_Re{Re}.npy", observ.out)
    np.save(f"/mnt/hgfs/VMshare/mass_proper_{identifier}_{angle}deg_Re{Re}.npy", massrep.out)
    simulation.save_checkpoint(f"/mnt/hgfs/VMshare/Haus_proper_{identifier}_{angle}deg_Re{Re}_save")

    """
    Auswertungsideen:
    plt.figure()
    plt.plot(pPU[:, 1], pPU[:, 6])

    plt.figure()
    sp = np.fft.fft(pPU[:, 6])
    freq = np.fft.fftfreq(pPU[:, 6].shape[-1], d=pPU[2, 1]-pPU[1, 1])
    plt.plot(freq, sp.real**2 + sp.imag**2)
    """

