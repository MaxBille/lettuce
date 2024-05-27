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

    def __init__(self, resolution_x, resolution_y, resolution_z, reynolds_number, mach_number, lattice, char_length_lu, char_length_pu, char_velocity_pu, char_density_pu, area):
        self.mpiObject = lattice.mpiObject
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
        self._mask = np.zeros(shape=(self.resolution_x, self.resolution_y, self.resolution_z), dtype=bool)
        self.rgrid = lt.RegularGrid([resolution_x, resolution_y, resolution_z], self.units.characteristic_length_lu,
                                self.units.characteristic_length_pu, endpoint=False, mpiObject=self.mpiObject)

    @property
    def grid(self):
        return self.rgrid()

    @property
    def mask(self):
        return self._mask

    @property
    def mask_global(self):
        return self._mask_global

    @mask.setter
    def mask(self, m):
        assert isinstance(m, np.ndarray) and m.shape == self.rgrid.global_shape
        self._mask_global = m.astype(bool)
        self._mask = self.rgrid.select(self.mask, rank=self.mpiObject.rank).astype(bool)

    def initial_solution(self, x):
        p = np.zeros_like(x[0], dtype=float)[None, ...]
        #u_char = np.array([self.units.characteristic_velocity_pu, 0.0, 0.0])[..., None, None, None]
        #u = (1 - self.grid.select(self.mask.astype(np.float), self.rank)) * u_char
        u = np.zeros((len(x),) + x[0].shape)
        if x[0].shape == self.rgrid.global_shape:
            u[0] = self.wind_speed_profile(np.where(self.mask_global, 0, x[2]), self.units.characteristic_velocity_pu, 0.25)
        elif x[0].shape == self.rgrid.shape:
            u[0] = self.wind_speed_profile(np.where(self.mask, 0, x[2]), self.units.characteristic_velocity_pu, 0.25)
        return p, u

    @property
    def boundaries(self): # inlet currently added via switch case based on call parameters!!!
        x, y, z = self.rgrid.global_grid()
        p, u = self.initial_solution(self.rgrid.global_grid())
        b = [
            lt.BounceBackBoundary(self.rgrid.select(self.mask_global | (z < 1e-6), rank=self.mpiObject.rank), self.units.lattice),
            lt.ZeroGradientOutlet(self.units.lattice, [0, 0, 1]),
            lt.EquilibriumOutletP(self.units.lattice, [1, 0, 0]),
            lt.NonEquilibriumExtrapolationInletU(self.units.lattice, self.units, [-1, 0, 0], np.array(u))
        ]
        return b

    def house3(self, o, eg_x_length, eg_y_length, roof_height, roof_overhang, roof_width=0, angle=45):
        #In LU, weil da sonst immer alles nicht stimmt^^ alles außer Winkel ist jetzt in LU roof_width gibts nich
        angle = angle * np.pi / 180
        eg_height = int(round(roof_height - (eg_x_length / 2 + roof_overhang) * np.tan(angle)))
        self.eg_height = eg_height
        self.roof_height = roof_height
        inside_mask = np.zeros_like(self.rgrid.global_grid()[0], dtype=bool)
        inside_mask[int(o[0]-eg_x_length/2):int(o[0]+eg_x_length/2), int(o[1]-eg_y_length/2):int(o[1]+eg_y_length/2), :eg_height] = True
        inside_mask[int(o[0]-eg_x_length/2-roof_overhang):int(o[0]+eg_x_length/2+roof_overhang), int(o[1]-eg_y_length/2):int(o[1]+eg_y_length/2), eg_height] = True
        inside_mask[int(o[0] - eg_x_length / 2-roof_overhang):int(o[0] + eg_x_length / 2+roof_overhang), int(o[1] - eg_y_length / 2):int(o[1] + eg_y_length / 2), eg_height+1:] = \
            np.where(self.units.convert_length_to_lu(self.rgrid.global_grid()[2][int(o[0] - eg_x_length / 2-roof_overhang):int(o[0] + eg_x_length / 2+roof_overhang), int(o[1] - eg_y_length / 2):int(o[1] + eg_y_length / 2),
        eg_height+1:]) < o[2] + roof_height + 0.5 - np.tan(angle) * np.abs(self.units.convert_length_to_lu(self.rgrid.global_grid()[0][int(o[0] - eg_x_length / 2-roof_overhang):int(o[0] + eg_x_length / 2+roof_overhang), int(o[1] - eg_y_length / 2):int(o[1] + eg_y_length / 2),
        eg_height+1:]) - o[0]), True, inside_mask[int(o[0] - eg_x_length / 2-roof_overhang):int(o[0] + eg_x_length / 2+roof_overhang), int(o[1] - eg_y_length / 2):int(o[1] + eg_y_length / 2), eg_height+1:])

        x1 = np.linspace(-0.1, 1.0, 12, endpoint=True)
        y1 = np.linspace(-0.5, 0.5, 11, endpoint=True)
        xy = np.meshgrid(y1, x1)
        # xy[x/y, x, y]
        roof_overhang = 6
        measurement_points = np.asarray([(eg_x_length / 2 + roof_overhang) * xy[1], eg_y_length * xy[0], np.zeros_like(xy[0])])

        meas_points = []
        meas_points2 = []
        meas_points3 = []
        meas_points4 = []

        for y in range(0, len(y1)):
            for x in range(0, len(x1)):
                measurement_points[:, x, y] = measurement_points[:, x, y] + o
                measurement_points[2, x, y] = roof_height + 0.001 - np.tan(angle) * np.abs(measurement_points[0, x, y] - o[0])
                measurement_points[:, x, y] = measurement_points[:, x, y].astype(np.int32)
                meas_points.append(measurement_points[:, x, y].astype(np.int32))
                meas_points2.append(measurement_points[:, x, y] + np.asarray([0, 0, 1]))
                meas_points3.append(measurement_points[:, x, y] + np.asarray([0, 0, 2]))
                meas_points4.append(measurement_points[:, x, y] + np.asarray([0, 0, 3]))

        measurement_points = meas_points + meas_points2 + meas_points3  + meas_points4

        middle_front = np.asarray([(eg_x_length / 2 + roof_overhang) * (-34/36), 0, roof_height + 0.001 - np.tan(angle) * np.abs((eg_x_length / 2 + roof_overhang) * -0.25)]) + np.asarray(o)
        for i in range(0, 3):
            measurement_points.append(middle_front.astype(np.int32) + np.asarray([0, 0, i]))
        # Hinter dem Einlass
        measurement_points.append(np.asarray([int(self.units.convert_length_to_lu(x)) for x in
                                              [self.rgrid.shape_pu[0] * 0.05, self.rgrid.shape_pu[1] / 2,
                                               self.rgrid.shape_pu[2] / 2]]))
        return inside_mask, measurement_points


    def wind_speed_profile(self, z, u_max, alpha=0.25):
        # based on DIN Onstwaswindprofil
        # all inputs in PU
        # return 0.77 * u_max * (z / 10) ** (alpha)
        return u_max * (z / self.roof_height) ** alpha

def run(device, mpiObject):
    Re = 20000#int(sys.argv[1])
    angle = 20#float(sys.argv[2])
    Ma = 0.1#float(sys.argv[3])
    identifier = "bla"#sys.argv[4]
    house_length_lu = 60
    house_length = 10

    print("Starting at: ", datetime.now())
    lattice = lt.Lattice(lt.D3Q27, device=device, dtype=torch.float32, MPIObject=mpiObject)
    lattice.equilibrium = lt.QuadraticEquilibrium_LessMemory(lattice)
    viscosity = 14.852989758837 * 10**(-6) # bei 15°C und 1 bar
    char_velocity = Re * viscosity / house_length
    flow = HouseFlow3D(360, 240, 180, Re, Ma, lattice, char_length_lu=house_length_lu, char_length_pu=house_length, char_density_pu=1.2250, char_velocity_pu=char_velocity, area=house_length_lu**2)
    #flow.mask, points = flow.house2([flow.units.convert_length_to_pu(int(a * b)) for a, b in zip([1/3, 0.5, 0], flow.grid.shape)], house_length, house_length, house_length*1.25, house_length*0.15, angle=angle)
    flow.mask, points = flow.house3([120,  120, 0], house_length_lu, house_length_lu, house_length_lu*1.25, 0, angle=angle)
    collision = lt.KBCCollision3D(lattice, tau=flow.units.relaxation_parameter_lu)
    streaming = lt.StandardStreaming(lattice)
    simulation = lt.Simulation(flow, lattice, collision, streaming, nan_steps=61)
    simulation.initialize_f_neq()
    #vtkrep = lt.VTKReporter(lattice, flow, interval=500, filename_base=f"/scratch/mkliem3s/final/Haus_final_{identifier}_{angle}deg_Re{Re}", nan_out=True)
    #simulation.reporters.append(vtkrep)
    #vtkrepP = lt.VTKReporterP(lattice, flow, interval=125, filename_base=f"/scratch/mkliem3s/final/Haus_p_final_{identifier}_{angle}deg_Re{Re}", nan_out=True)
    #simulation.reporters.append(vtkrepP)
    #massRep = lt.Mass(lattice, flow)

    #vtkrep.output_mask(simulation.no_collision_mask)

    print(f"Parameters: Re: {Re}, Ma {Ma}, House_length_LU: {house_length_lu} Identifier: {identifier}")
    print(f"Relaxation parameter: {simulation.flow.units.relaxation_parameter_lu}")
    print(f"Duration of a time step in PU: {simulation.flow.units.convert_time_to_pu(1)}")
    print(f"Characteristic velocity: {char_velocity}")

    time_start = time.time()
    print('MLUPS: ', simulation.step(4 * 10**4))
    time_end = time.time()

    print("The simulation took {} hours.".format((time_end - time_start) / 3600))

if __name__ == "__main__":
    device = torch.device("cuda")
    pcList=[[0,20],[1,40]]
    #gpuList_siegen=[[4,"gpu-node001"],[4,"gpu-node002"],[4,"gpu-node003"],[4,"gpu-node004"],[1,"gpu-node005"],[1,"gpu-node006"],[1,"gpu-node007"],[1,"gpu-node008"],[2,"gpu-node009"],[2,"gpu-node010"]]
    gpuList_hbrs=[[4, "wr15"], [1, "wr12"], [1, "wr16"], [1, "wr17"], [1, "wr18"], [1, "wr19"]]
    mpiOBJ=lt.mpiObject(1, sizeList=pcList, gpuList=gpuList_hbrs, setParts=0, gridRefinment=0, printUserInfo=1)
    lt.running(run, device, mpiOBJ)

