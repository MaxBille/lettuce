import numpy as np
from lettuce.unit import UnitConversion
from lettuce.util import append_axes
from lettuce.boundary import EquilibriumBoundaryPU, BounceBackBoundary, HalfwayBounceBackBoundary, FullwayBounceBackBoundary, EquilibriumOutletP, AntiBounceBackOutlet
    # EquilibriumInletPU,



class ObstacleMax:
    """
        Flow class to simulate the flow around an object (mask).
        It consists of one inflow (equilibrium boundary)
        and one outflow (anti-bounce-back-boundary), leading to a flow in positive x direction.

        Parameters
        ----------


        Attributes
        ----------
        obstacle_mask : np.array with dtype = np.bool
            Boolean mask to define the obstacle. The shape of this object is the shape of the grid.
            Initially set to zero (no obstacle).

        Examples
        --------
        Initialization of flow around a cylinder:

        >>> from lettuce import Lattice, D2Q9
        >>> flow = ObstacleMax(#parameters)
        >>> radius = 0.5 * flow.char_length_pu
        >>> x_pos = 0.25 * flow.domain_length_cl
        >>> y_pos = 0.5 * flow.domain_width_cl
        >>> x, y = flow.grid
        >>> condition = np.sqrt((x - x_pos) ** 2 + (y - y_pos) ** 2) < radius  # (x-x0)²+(y-y0)²<r² in PU
        >>> flow.solid_mask[np.where(condition)] = 1
        >>> flow.obstacle_mask[np.where(condition)] = 1
    """

    def __init__(self, reynolds_number, mach_number, lattice, char_length_pu, char_length_lu, char_velocity_pu=1, y_cl=5, x_cl=10, lateral_walls=True, hwbb=True, perturb_init=True, u_init=0):
        self.shape = (x_cl * char_length_lu, y_cl * char_length_lu)  # shape of the domain
        self.char_length_pu = char_length_pu
        self.domain_length_cl = x_cl
        self.domain_width_cl = y_cl

        self.perturb_init = perturb_init
        self.u_init = u_init
        self.lateral_walls = lateral_walls
        self.hwbb = hwbb

        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number,
            mach_number=mach_number,
            characteristic_length_lu=char_length_lu,
            characteristic_length_pu=char_length_pu,
            characteristic_velocity_pu=char_velocity_pu  ### reminder: u_char_lu = Ma * cs_lu = Ma * 1/sqrt(3)
        )

        self.solid_mask = np.zeros(shape=self.shape, dtype=bool)  # marks all solid nodes (obstacle, walls, ...)
        self.in_mask = np.zeros(self.grid[0].shape, dtype=bool)
        self.in_mask[0, 1:-1] = True
        self.wall_mask = np.zeros_like(self.solid_mask)
        if self.lateral_walls:
            self.wall_mask[:, [0, -1]] = True
            self.solid_mask[np.where(self.wall_mask)] = 1
        self._obstacle_mask = np.zeros_like(self.solid_mask)  # marks all obstacle nodes (for fluid-solid-force_calc)

        # generate parabolic velocity profile for inlet BC if lateral_walls=True (== channel-flow)
        self.u_inlet = self.units.characteristic_velocity_pu * self._unit_vector()
        if self.lateral_walls:
            ## Parabelförmige Geschwindigkeit, vom zweiten bis vorletzten Randpunkt (keine Interferenz mit lateralen Wänden (BBB  oder periodic))
            ## How to Parabel:
            ## 1.Parabel in Nullstellenform: y = (x-x1)*(x-x2)
            ## 2.nach oben geöffnete Parabel mit Nullstelle bei x1=0 und x2=x0: y=-x*(x-x0)
            ## 3.skaliere Parabel, sodass der Scheitelpunkt immer bei ys=1.0 ist: y=-x*(x-x0)*(1/(x0/2)²)
            ## (4. optional) skaliere Amplitude der Parabel mit 1.5, um dem Integral einer homogenen Einstromgeschwindigkeit zu entsprechen
            ny = self.shape[1]  # Gitterpunktzahl in y-Richtung
            ux_temp = np.zeros((1, ny))  # x-Geschwindigkeiten der Randbedingung
            y_coordinates = np.linspace(0, ny, ny)  # linspace() erzeugt n Punkte zwischen 0 und ny inklusive 0 und ny, so wird die Parabel auch symmetrisch und ist trotzdem an den Rändern NULL
            ux_temp[:, 1:-1] = - np.array(self.u_inlet).max() * y_coordinates[1:-1] * (y_coordinates[1:-1] - ny) * 1 / (ny / 2) ** 2
            # Skalierungsfaktor 3/2=1.5 für die Parabelamplitude, falls man im Integral der Konstantgeschwindigkeit entsprechenn möchte.
            # in 2D braucht u1 dann die Dimension 1 x ny (!)
            uy_temp = np.zeros_like(ux_temp)  # y-Geschwindigkeit = 0
            self.u_inlet = np.stack([ux_temp, uy_temp], axis=0)  # verpacke u-Feld

    @property
    def obstacle_mask(self):
        return self._obstacle_mask

    @obstacle_mask.setter
    def obstacle_mask(self, m):
        assert isinstance(m, np.ndarray) and m.shape == self.shape
        self._obstacle_mask = m.astype(bool)
        self.solid_mask[np.where(self.obstacle_mask)] = 1

    def initial_solution(self, x):
        p = np.zeros_like(x[0], dtype=float)[None, ...]
        u_max_lu = self.units.characteristic_velocity_lu * self._unit_vector()
        u_max_lu = append_axes(u_max_lu, self.units.lattice.D)
        self.solid_mask[np.where(self.obstacle_mask)] = 1  # kann vielleicht raus, da die Belegung über den obstacle_mask.setter ausreichen sollte
        ### initial velocity field: "u_init"-parameter
        # 0: uniform u=0
        # 1: uniform u=1
        # 2: parabolic, amplitude u_char_lu (similar to poiseuille-flow)
        u = (1 - self.solid_mask) * u_max_lu
        if self.u_init == 1:
            # initiale velocity u=1 on every fluid node
            u = (1 - self.solid_mask) * u_max_lu
            print("initial u=0")
        elif self.u_init == 2:  # parabolic along y, uniform along x (similar to poiseuille-flow)
            ny = self.shape[1]  # Gitterpunktzahl in y-Richtung
            ux_factor = np.zeros(ny)  # Vektor für erste Spalte (u(x=0)) des u-Feldes
            # Geschwindigkeitsparabel mit Amplitude 1 (zur Aufmultiplikation auf das gesamte Geschwindigkeitsfeld, s.u.)
            y_coordinates = np.linspace(0, ny, ny)
            ux_factor[1:-1] = - y_coordinates[1:-1] * (y_coordinates[1:-1] - ny) * 1 / (ny / 2) ** 2
            u = np.einsum('k,ijk->ijk', ux_factor, u)
        else:
            u = u*0  # uniform u=0

        ### perturb initial velocity field-symmetry to trigger 'von Karman' vortex street
        # perturb_init = True/False
        if self.perturb_init:
            # overlays a sine-wave on the second row of nodes
            ny = x[1].shape[1]
            if u.max() < 0.5 * self.units.characteristic_velocity_lu:
                # add perturbation for small velocities
                u[0][1] += np.sin(np.arange(0, ny) / ny * 2 * np.pi) * self.units.characteristic_velocity_lu * 1.0
            else:
                # multiply scaled down perturbation
                u[0][1] *= 1 + np.sin(np.arange(0, ny) / ny * 2 * np.pi) * 0.3
        return p, u

    @property
    def grid(self):
        xyz = tuple(self.units.convert_length_to_pu(np.linspace(0,n,n)) for n in self.shape)  # Tupel aus Listen der x-Wert, y-Werte, (und z-Werte)
        return np.meshgrid(*xyz, indexing='ij')  # meshgrid aus den x-, y- (und z-)Werten

    @property
    def boundaries(self):
        # inlet ("left side", x[0],y[1:-1])
        inlet_boundary = EquilibriumBoundaryPU(
                                            self.in_mask,
                                            self.units.lattice, self.units,
                                            #self.units.characteristic_velocity_pu * self._unit_vector())
                                            self.u_inlet) # übergibt 1 x D Vektor oder ny x D Vektor, beides funktioniert dank Einsum in EquilibriumBoundaryPU
        # lateral walls ("top and bottom walls", x[:], y[0,-1])
        lateral_boundary = None
        if self.lateral_walls:
            if self.hwbb:  # use halfway bounce back
                lateral_boundary = HalfwayBounceBackBoundary(self.wall_mask, self.units.lattice)
            else:  # else use fullway bounce back
                lateral_boundary = FullwayBounceBackBoundary(self.wall_mask, self.units.lattice)
        # outlet ("right side", x[-1],y[:])
        outlet_boundary = EquilibriumOutletP(self.units.lattice, [1, 0])  # Auslass in positive x-Richtung
        # obstacle (obstacle "cylinder" with radius centered at position x_pos, y_pos)
        obstacle_boundary = None
        if self.hwbb:
            obstacle_boundary = HalfwayBounceBackBoundary(self.obstacle_mask, self.units.lattice)
        else:
            obstacle_boundary = FullwayBounceBackBoundary(self.obstacle_mask, self.units.lattice)
        if lateral_boundary is None:  # if lateral boundary is periodic...don't return a boundary-object
            return [
                inlet_boundary,
                outlet_boundary,
                obstacle_boundary
                ]
        else:
            return [
                inlet_boundary,
                outlet_boundary,
                lateral_boundary,
                obstacle_boundary
                ]

    def _unit_vector(self, i=0):
        return np.eye(self.units.lattice.D)[i]