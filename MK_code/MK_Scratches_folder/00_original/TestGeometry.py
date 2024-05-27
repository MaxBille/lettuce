import Geometry3D as gd
import lettuce as lt
import numpy as np
import torch

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

    def house4(self, o, eg_x_length, eg_y_length, roof_height, roof_overhang, roof_width=0, angle=45):
        angle = angle * np.pi / 180
        eg_height = int(round(roof_height - (eg_x_length / 2 + roof_overhang) * np.tan(angle)))

        ridge_0 = gd.Point(eg_x_length / 2 + roof_overhang, 0, roof_height)
        ridge_1 = gd.Point(eg_x_length / 2 + roof_overhang, eg_y_length, roof_height)

        roof_corner_0 = gd.Point(0, 0, eg_height)
        roof_corner_1 = gd.Point(0, eg_y_length, eg_height)
        roof_corner_2 = gd.Point(eg_x_length + 2 * roof_overhang, eg_y_length, eg_height)
        roof_corner_3 = gd.Point(eg_x_length + 2 * roof_overhang, 0, eg_height)

        eg_corner_0_up = gd.Point(roof_overhang, 0, eg_height)
        eg_corner_1_up = gd.Point(roof_overhang, eg_y_length, eg_height)
        eg_corner_2_up = gd.Point(eg_x_length + roof_overhang, eg_y_length, eg_height)
        eg_corner_3_up = gd.Point(eg_x_length + roof_overhang, 0, eg_height)

        eg_corner_0_down = gd.Point(roof_overhang, 0, 0)
        eg_corner_1_down = gd.Point(roof_overhang, eg_y_length, 0)
        eg_corner_2_down = gd.Point(eg_x_length + roof_overhang, eg_y_length, 0)
        eg_corner_3_down = gd.Point(eg_x_length + roof_overhang, 0, 0)

        r = gd.Renderer()
        r.add((ridge_0, 'r', 1))
        r.add((ridge_1, 'r', 1))

        r.add((roof_corner_0, 'r', 1))
        r.add((roof_corner_1, 'r', 1))
        r.add((roof_corner_2, 'r', 1))
        r.add((roof_corner_3, 'r', 1))

        r.add((eg_corner_0_up, 'r', 1))
        r.add((eg_corner_1_up, 'r', 1))
        r.add((eg_corner_2_up, 'r', 1))
        r.add((eg_corner_3_up, 'r', 1))

        r.add((eg_corner_0_down, 'r', 1))
        r.add((eg_corner_1_down, 'r', 1))
        r.add((eg_corner_2_down, 'r', 1))
        r.add((eg_corner_3_down, 'r', 1))

        wxn = gd.ConvexPolygon((eg_corner_0_up, eg_corner_0_down, eg_corner_1_down, eg_corner_1_up))
        wyp = gd.ConvexPolygon((eg_corner_2_up, eg_corner_2_down, eg_corner_1_down, eg_corner_1_up))
        wxp = gd.ConvexPolygon((eg_corner_2_up, eg_corner_2_down, eg_corner_3_down, eg_corner_3_up))
        wyn = gd.ConvexPolygon((eg_corner_0_up, eg_corner_0_down, eg_corner_3_down, eg_corner_3_up))
        boden = gd.ConvexPolygon((eg_corner_0_down, eg_corner_1_down, eg_corner_2_down, eg_corner_3_down))
        decke = gd.ConvexPolygon((eg_corner_0_up, eg_corner_1_up, eg_corner_2_up, eg_corner_3_up))

        eg = gd.ConvexPolyhedron((wxn, wyp, wxp, wyn, boden, decke))

        boden = gd.ConvexPolygon((roof_corner_0, roof_corner_1, roof_corner_2, roof_corner_3))
        dfn = gd.ConvexPolygon((roof_corner_0, roof_corner_1, ridge_0, ridge_1))
        dfp = gd.ConvexPolygon((roof_corner_2, roof_corner_3, ridge_0, ridge_1))
        sp = gd.ConvexPolygon((roof_corner_0, roof_corner_3, ridge_0))
        sn = gd.ConvexPolygon((roof_corner_1, roof_corner_2, ridge_1))

        dach = gd.ConvexPolyhedron((boden, dfp, dfn, sp, sn))
        r.add((eg, 'g', 1))
        r.add((dach, 'g', 1))
        r.show()


        o = gd.Vector(o)
        dach.move((o))
        eg.move((o))

        inside_mask = np.zeros_like(self.rgrid.global_grid()[0], dtype=bool)

        for x in range(0, self.rgrid.global_shape[0]):
            print(x)
            for y in range(0, self.rgrid.global_shape[1]):
                for z in range(0, self.rgrid.global_shape[2]):
                    point = gd.Point((x, y, z))
                    if dach.__contains__(point) or eg.__contains__(point):
                        inside_mask[x,y,z] = True
        print("blub")

Re = 20000#int(sys.argv[1])
angle = 20#float(sys.argv[2])
Ma = 0.1#float(sys.argv[3])
identifier = "bla"#sys.argv[4]
house_length_lu = 60
house_length = 10

lattice = lt.Lattice(lt.D3Q27, device=torch.device("cpu"), dtype=torch.float32)
lattice.equilibrium = lt.QuadraticEquilibrium_LessMemory(lattice)
viscosity = 14.852989758837 * 10**(-6) # bei 15°C und 1 bar
char_velocity = Re * viscosity / house_length
flow = HouseFlow3D(360, 240, 180, Re, Ma, lattice, char_length_lu=house_length_lu, char_length_pu=house_length, char_density_pu=1.2250, char_velocity_pu=char_velocity, area=house_length_lu**2)

flow.house4([120, 120, 0], 60, 60, 72, 6, 0, angle=35)