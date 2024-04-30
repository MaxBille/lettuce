import lettuce as lt
import numpy as np
import torch
import sys

class HouseFlow3D(object):

    def __init__(self, resolution_x, resolution_y, resolution_z,
                 reynolds_number, mach_number, lattice,
                 char_length_lu, char_length_pu,
                 char_velocity_pu, char_density_pu):
        """
            Flow class to simulate the flow around an object (mask) in 3D.
            Parameters:
            resolution_x, resolution_y, resolution_z: domain resolutions, in LU
            lattice: object of the class with the same name from lettuce
            char_length_lu: length of the base of the house, in LU
            char_length_pu: length of the base of the house, in PU
            char_velocity_pu: characteristic velocity (inlet velocity), in PU
            char_density_pu: characteristic density, in PU
        """
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.resolution_z = resolution_z
        self.units = lt.UnitConversion(
            lattice,
            reynolds_number=reynolds_number,
            mach_number=mach_number,
            characteristic_length_lu=char_length_lu,
            characteristic_length_pu=char_length_pu,
            characteristic_velocity_pu=char_velocity_pu,
            characteristic_density_pu=char_density_pu,
            characteristic_density_lu=1
        )
        self._mask = np.zeros(shape=(self.resolution_x, self.resolution_y,
                                     self.resolution_z), dtype=np.bool)

        self.grid = lt.RegularGrid([resolution_x, resolution_y, resolution_z],
                                    self.units.characteristic_length_lu,
                                    self.units.characteristic_length_pu)

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, m):
        assert isinstance(m, np.ndarray) and m.shape == self.grid.global_shape
        self._mask = m.astype(np.bool)

    def initial_solution(self, x):
        """Returns the initial macroscopic values (u, rho) at each lattice node
            Initialises speed using the velocity profile function and
            pressure difference from reference pressure as 0

            Inputs:
            x: grid, in LU
        """
        p = np.zeros_like(x[0], dtype=float)[None, ...]
        u = np.zeros((len(x),) + x[0].shape)
        u[0] = self.wind_speed_profile(np.where(self.mask, 0, x[2]),
                            self.units.characteristic_velocity_pu, 0.25)
        return p, u

    @property
    def boundaries(self):
        """Returns the objects of each boundary class for use by the Simulation class
        """
        x, y, z = self.grid.global_grid()
        p, u = self.initial_solution(self.grid())
        return [
            lt.BounceBackBoundary(self.mask | (z < 1e-6), self.units.lattice),
            lt.ZeroGradientOutlet(self.units.lattice, [0, 0, 1]),
            lt.EquilibriumOutletP(self.units.lattice, [1, 0, 0]),
            lt.NonEquilibriumExtrapolationInletU(self.units.lattice,
                                        self.units, [-1, 0, 0], np.array(u))
        ]


    def house(self, o, eg_x_length, eg_y_length,
                                roof_height, roof_overhang, angle=35):
        """Outputs mask for flow: house with square base
            and gable-roof parallel to y direction and overhang

            Inputs:
            eg_x_length: length of the house base in x-direction, in LU
            eg_y_length: length of the house base in y-direction, in LU
            roof_height: height of the top of the roof, in LU

        """
        angle = angle * np.pi / 180
        eg_height = int(round(roof_height - (eg_x_length / 2 + roof_overhang)
                              * np.tan(angle)))
        self.roof_height = roof_height
        inside_mask = np.zeros_like(self.grid.global_grid()[0], dtype=bool)
        inside_mask[int(o[0]-eg_x_length/2):int(o[0]+eg_x_length/2),
                    int(o[1]-eg_y_length/2):int(o[1]+eg_y_length/2),
                    :eg_height] = True
        inside_mask[int(o[0]-eg_x_length/2-roof_overhang):
                    int(o[0]+eg_x_length/2+roof_overhang),
                    int(o[1]-eg_y_length/2):
                    int(o[1]+eg_y_length/2), eg_height] = True
        inside_mask[int(o[0] - eg_x_length / 2-roof_overhang):
                    int(o[0] + eg_x_length / 2+roof_overhang),
                    int(o[1] - eg_y_length / 2):
                    int(o[1] + eg_y_length / 2), eg_height+1:] = \
            np.where(self.units.convert_length_to_lu(
                self.grid()[2][int(o[0] - eg_x_length / 2-roof_overhang):
                               int(o[0] + eg_x_length / 2+roof_overhang),
                               int(o[1] - eg_y_length / 2):
                               int(o[1] + eg_y_length / 2), eg_height+1:]) <
                o[2] + roof_height + 0.5 - np.tan(angle) *
                    np.abs(self.units.convert_length_to_lu(
                            self.grid()[0][
                            int(o[0] - eg_x_length / 2-roof_overhang):
                            int(o[0] + eg_x_length / 2+roof_overhang),
                            int(o[1] - eg_y_length / 2):
                            int(o[1] + eg_y_length / 2),
                            eg_height+1:]) - o[0]),
                True,
                inside_mask[int(o[0] - eg_x_length / 2-roof_overhang):
                            int(o[0] + eg_x_length / 2+roof_overhang),
                            int(o[1] - eg_y_length / 2):
                            int(o[1] + eg_y_length / 2), eg_height+1:])
        return inside_mask


    def wind_speed_profile(self, z, u_0, alpha=0.25):
        """Returns the objects of each boundary class for use by the Simulation class

            Inputs:
            z: height value or array of height values, in PU
            u_0: characteristic velocity, applies at roof_height, in PU
            alpha: profile shape exponent
        """
        return u_0 * (z / self.roof_height) ** alpha


# read angle from context of program start
angle = float(sys.argv[1])
# calculate / define necessary values for description of flow
Re = 20000
house_length_lu = 60
house_length = 10
viscosity = 14.852989758837 * 10**(-6)
char_velocity = Re * viscosity / house_length


lattice = lt.Lattice(lt.D3Q27, device=torch.device("cuda"), dtype=torch.float32)
lattice.equilibrium = lt.QuadraticEquilibrium_LessMemory(lattice)
# supply flow object with all values to describe the flow
flow = HouseFlow3D(360, 240, 180, Re, mach_number=0.1, lattice=lattice,
                   char_length_lu=house_length_lu, char_length_pu=house_length,
                   char_density_pu=1.2250, char_velocity_pu=char_velocity)
# generate house mask and set it as mask of flow
flow.mask, points = flow.house([120,  120, 0], eg_x_length=house_length_lu,
                               eg_y_length=house_length_lu,
                               roof_height=house_length_lu*1.25,
                               roof_overhang=6, angle=angle)
collision = lt.KBCCollision3D(lattice, tau=flow.units.relaxation_parameter_lu)
streaming = lt.StandardStreaming(lattice)
simulation = lt.Simulation(flow, lattice, collision, streaming)

# add reporters to output results
vtkRep = lt.VTKReporter(lattice, flow, interval=500, filename_base=f"Haus{angle}")
simulation.reporters.append(vtkRep)
vtkRep.output_mask(simulation.no_collision_mask)

vtkRepP = lt.VTKReporterP(lattice, flow, interval=125, filename_base=f"Haus_p{angle}")
simulation.reporters.append(vtkRepP)

# perform the simulation
print('MLUPS: ', simulation.step(4 * 10**5))