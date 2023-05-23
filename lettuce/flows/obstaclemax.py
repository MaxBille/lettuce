import numpy as np
from lettuce.unit import UnitConversion
from lettuce.util import append_axes
from lettuce.boundary import EquilibriumBoundaryPU, BounceBackBoundary, HalfwayBounceBackBoundary, FullwayBounceBackBoundary, EquilibriumOutletP, AntiBounceBackOutlet
    # EquilibriumInletPU,



class ObstacleMax:
    """
        (!) refined version of flow/obstacle.py (so far only for 2D obstacles)
        Flow class to simulate the flow around an object (obstacle_mask).
        It consists of one inflow (equilibrium boundary)
        and one outflow (anti-bounce-back-boundary), leading to a flow in positive x direction.

        Parameters (selection)
        ----------
        lateral_walls: True = channel flow with parabolic velocity inlet, False = periodic lateral boundaries and uniform velocity inlet
        hwbb: True = use halfway bounce back boundary, False = use fullway bounce back boundary
        perturb_init: True = slight perturbation in the initial condition to trigger vKarman vortex street in symmetrical systems (for Re>46)
        u_init: initial velocity profile: 0 = uniform u=0, 1 = uniform u=u_char, 2 = parabolic (poiseuille like)

        Attributes (selection)
        ----------
        obstacle_mask : np.array with dtype = np.bool
            Boolean mask to define the obstacle. The shape of this object is the shape of the grid.
            Initially set to zero (no obstacle).

    """

    def __init__(self, reynolds_number, mach_number, lattice, char_length_pu, char_length_lu, char_velocity_pu=1, y_lu=5, x_lu=10, lateral_walls=True, hwbb=True, perturb_init=True, u_init=0):
        self.shape = (int(x_lu), int(y_lu))  # shape of the domain in LU
        self.char_length_pu = char_length_pu  # characteristic length
        self.x_lu = x_lu  # domain length
        self.y_lu = y_lu  # domain width

        self.perturb_init = perturb_init  # toggle: introduce asymmetry in initial solution
        self.u_init = u_init  # toggle: initial solution velocity profile
        self.lateral_walls = lateral_walls  # toggle: lateral walls or lateral periodic boundary
        self.hwbb = hwbb  # toggle: bounce back algorithm: halfway or fullway

        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number,
            mach_number=mach_number,
            characteristic_length_lu=char_length_lu,
            characteristic_length_pu=char_length_pu,
            characteristic_velocity_pu=char_velocity_pu  ### reminder: u_char_lu = Ma * cs_lu = Ma * 1/sqrt(3)
        )

        self.solid_mask = np.zeros(shape=self.shape, dtype=bool)  # marks all solid nodes (obstacle, walls, ...)
        self.in_mask = np.zeros(self.grid[0].shape, dtype=bool)  # marks all inlet nodes
        self.in_mask[0, 1:-1] = True  # inlet on the left
        self.wall_mask = np.zeros_like(self.solid_mask)  # marks lateral walls
        if self.lateral_walls:
            self.wall_mask[:, [0, -1]] = True
            self.solid_mask[np.where(self.wall_mask)] = 1
        self._obstacle_mask = np.zeros_like(self.solid_mask)  # marks all obstacle nodes (for fluid-solid-force_calc.)

        # generate parabolic velocity profile for inlet BC if lateral_walls=True (== channel-flow)
        self.u_inlet = self.units.characteristic_velocity_pu * self._unit_vector()
        if self.lateral_walls:
            ## parabolic velocity profile, zeroing on the edges
            ## How to parabola:
            ## 1.parabola in factoriezed form (GER: "Nullstellenform"): y = (x-x1)*(x-x2)
            ## 2.parabola with a maximum and zero at x1=0 und x2=x0: y=-x*(x-x0)
            ## 3.scale parabola, to make y_s(x_s)=1 the maximum: y=-x*(x-x0)*(1/(x0/2)²)
            ## (4. optional) scale amplitude with 1.5 to have a mean velocity of 1, also making the integral of a homogeneous velocity profile with u=1 and the parabolic profile being equal
            ny = self.shape[1]  # number of gridpoints in y direction
            ux_temp = np.zeros((1, ny))  # variable for x-velocities on inlet boundary
            y_coordinates = np.linspace(0, ny, ny)  # linspace() creates n points between 0 and ny, including 0 and ny:
                # top and bottom velocity values will be zero to agree with wall-boundary-condition
            ux_temp[:, 1:-1] = - np.array(self.u_inlet).max() * y_coordinates[1:-1] * (y_coordinates[1:-1] - ny) * 1 / (ny / 2) ** 2  # paraboplic velocity profile
            # scale with 1.5 to achieve a mean velocity of u_char!
            # in 2D u1 needs Dimension 1 x ny (!)
            uy_temp = np.zeros_like(ux_temp)  # y-velocities = 0
            self.u_inlet = np.stack([ux_temp, uy_temp], axis=0)  # stack/pack u-field

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
        self.solid_mask[np.where(self.obstacle_mask)] = 1  # is this line still needed? obstacle_mask.setter should add obstacle_mask to solid_mask
        ### initial velocity field: "u_init"-parameter
        # 0: uniform u=0
        # 1: uniform u=1
        # 2: parabolic, amplitude u_char_lu (similar to poiseuille-flow)
        u = (1 - self.solid_mask) * u_max_lu
        if self.u_init == 1:
            # initiale velocity u=1 on every fluid node
            u = (1 - self.solid_mask) * u_max_lu
        elif self.u_init == 2:  # parabolic along y, uniform along x (similar to poiseuille-flow)
            ny = self.shape[1]  # number of gridpoints in y direction
            ux_factor = np.zeros(ny)  # vector for one column (u(x=0))
            # multiply parabolic profile with every column of the velocity field:
            y_coordinates = np.linspace(0, ny, ny)
            ux_factor[1:-1] = - y_coordinates[1:-1] * (y_coordinates[1:-1] - ny) * 1 / (ny / 2) ** 2
            u = np.einsum('k,ijk->ijk', ux_factor, u)
        else:
            u = u*0  # uniform u=0

        ### perturb initial velocity field-symmetry to trigger 'von Karman' vortex street
        # perturb_init = True/False
        if self.perturb_init:
            # overlays a sine-wave on the second column of nodes
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
        xyz = tuple(self.units.convert_length_to_pu(np.linspace(0,n,n)) for n in self.shape)  # tuple of lists of x,y,(z)-values/indices
        return np.meshgrid(*xyz, indexing='ij')  # meshgrid of x-, y- (und z-)Wvalues/indices

    @property
    def boundaries(self):
        # inlet ("left side", x[0],y[1:-1])
        inlet_boundary = EquilibriumBoundaryPU(
                                            self.in_mask,
                                            self.units.lattice, self.units,
                                            #self.units.characteristic_velocity_pu * self._unit_vector())
                                            self.u_inlet) # works with a 1 x D vector or an ny x D vector thanks to einsum-magic in EquilibriumBoundaryPU
        # lateral walls ("top and bottom walls", x[:], y[0,-1])
        lateral_boundary = None
        if self.lateral_walls:
            if self.hwbb:  # use halfway bounce back
                lateral_boundary = HalfwayBounceBackBoundary(self.wall_mask, self.units.lattice)
            else:  # else use fullway bounce back
                lateral_boundary = FullwayBounceBackBoundary(self.wall_mask, self.units.lattice)
        # outlet ("right side", x[-1],y[:])
        outlet_boundary = EquilibriumOutletP(self.units.lattice, [1, 0])  # outlet in positive x-direction
        # obstacle (for example: obstacle "cylinder" with radius centered at position x_pos, y_pos) -> to be set via obstacle_mask.setter
        obstacle_boundary = None
        # (!) the obstacle_boundary should alway be the last boundary in the list of boundaries to correctly calculate forces on the obstacle
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
