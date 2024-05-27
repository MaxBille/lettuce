import lettuce as lt
import numpy as np
import torch
from matplotlib import  pyplot as plt

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
        u[0] = self.wind_speed_profile(np.where(self.mask, 0, x[2]), self.units.characteristic_velocity_pu, np.max(x[2]), 2)
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
        else:
            b.append(lt.NonEquilibriumExtrapolationInletU(self.units.lattice, self.units, [-1, 0, 0], np.array(u)))
        return b

    def wind_speed_profile(self, z, u_max, z_max, alpha):
        # based on wikipedia article about wind profile
        return u_max * (z / z_max) ** (1/alpha)

lattice = lt.Lattice(lt.D3Q27, device=torch.device("cpu"), dtype=torch.float32)

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

Re = 100
angle = 50
identifier = "ggwPUtest"
house_length_lu = 50
house_length = 15

flow = HouseFlow3D(350, 250, 175, Re, 0.1, lattice, char_length_lu=house_length_lu, char_length_pu=house_length, char_density_pu=10**5, char_velocity_pu=30, area=house_length_lu**2)

p = np.load("/home/martin/Masterthesis/Clusterinhalt/p/p_proper_ggwPUfinal_30deg_Re20000.npy")
p = np.hstack([p[:, :2], flow.units.convert_pressure_pu_to_density_lu(p[:, 2:])])

flow = HouseFlow3D(350, 250, 175, Re, 0.1, lattice, char_length_lu=house_length_lu, char_length_pu=house_length, char_density_pu=1.2250, char_velocity_pu=0.03, area=house_length_lu**2)
p = np.hstack([p[:, :2], flow.units.convert_density_lu_to_pressure_pu(p[:, 2:])])

plt.figure()
plt.plot(p[:, 1], p[:, 6])
#N=500
#test = np.convolve(p[:, 6], np.ones(N)/N, mode='valid')
#plt.plot(p[-len(test):, 1], test)
plt.plot(p[:, 1], p[:, 11])

plt.figure()
sp = np.fft.fft(p[:, 6])
freq = np.fft.fftfreq(p[:, 6].shape[-1], d=p[2, 1]-p[1, 1])
plt.plot(freq, sp.real**2 + sp.imag**2)
plt.show()