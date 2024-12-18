from math import ceil, floor, sqrt
import numpy as np
import torch
from lettuce.lattices import Lattice  # for TYPING purposes
from lettuce.unit import UnitConversion
from lettuce.util import append_axes
from lettuce.boundary import EquilibriumBoundaryPU, \
    BounceBackBoundary, HalfwayBounceBackBoundary, FullwayBounceBackBoundary, EquilibriumOutletP, AntiBounceBackOutlet, \
    InterpolatedBounceBackBoundary, InterpolatedBounceBackBoundary_compact_v1, InterpolatedBounceBackBoundary_compact_v2, InterpolatedBounceBackBoundary_occ, \
    SlipBoundary, FullwayBounceBackBoundary_compact, FullwayBounceBackBoundary_occ, HalfwayBounceBackBoundary_compact_v1, HalfwayBounceBackBoundary_compact_v2, HalfwayBounceBackBoundary_occ, \
    HalfwayBounceBackBoundary_compact_v3, PartiallySaturatedBoundary, RampedEquilibriumBoundaryPU
from lettuce.boundary_mk import NonEquilibriumExtrapolationInletU, SyntheticEddyInlet, ZeroGradientOutlet
from pspelt.obstacleFunctions import makeGrid
import time


# houseFlow3D by M.Kliemank, from MA-Thesis-CD-ROM "simulation_code.py"
class HouseFlow3D(object):
    shape: tuple[int, int, int] or tuple[int, int]

    def __init__(self, shape: tuple, reynolds_number: float, mach_number: float, lattice: Lattice,
                 domain_constraints: tuple, char_length_lu: float,
                 char_length_pu: float = 1, char_velocity_pu=1, u_init: 0 or 1 or 2 = 0,
                 reference_height_pu = 0,  # with respect to ground_height_PU!, PU-height, at which char_velocity_pu is in inflow-profile (relevant for WSP, see below)
                 ground_height_pu = 0,  # PU-height of groundlevel from which the building hight and WSP height is calculated against; basically the offset between PU-GRID-coordinates and PU-house-coordinates
                 inlet_bc: str = "eqin", outlet_bc: str = "eqoutp", ground_bc: str = "fwbb", house_bc: str = "fwbb",
                 top_bc: str = "zgo",
                 inlet_ramp_steps = 0,
                 house_solid_boundary_data = None,
                 ground_solid_boundary_data = None,
                 K_Factor=10,  # K_factor for SEI boundary inlet
                 L=3,  # L for SEI
                 N=34,  # N number of random vortices for SEI
                 wsp_shift_up_pu=0,  # how many PU to shift the u_inlet profile UP
                 wsp_y0=None,  # y0 up until which u=0 for wind speed profile (not a shift, but set of zero-height -> compresses profile)
                 wsp_alpha=0.25
                 ):
        # flow and boundary settings
        self.u_init = u_init  # toggle: initial solution velocity profile type
        self.lattice = lattice
        assert len(shape) == lattice.D
        self.shape = shape  # LU-indices
        self.ndim = lattice.D
        self.char_length_pu = char_length_pu  # characteristic length
        self.domain_constraints = domain_constraints  # ([xmin, ymin], [xmax, ymax]) if dim == 2 else ([xmin, ymin, zmin], [xmax, ymax, zmax])  # Koordinatensystem in PU, abh. von der stl und deren ursprung
        self.reference_height_pu = reference_height_pu
        self.ground_height_pu = ground_height_pu

        # wind speed profile parameters
        if wsp_y0 is None:
            self.wsp_y0 = ground_height_pu
        else:
            self.wsp_y0 = wsp_y0
        self.wsp_shift_up_pu = wsp_shift_up_pu
        self.wsp_alpha = wsp_alpha

        # bc_types
        self.inlet_bc = inlet_bc
        self.outlet_bc = outlet_bc
        self.ground_bc = ground_bc
        self.house_bc = house_bc
        self.top_bc = top_bc

        self.inlet_ramp_steps = inlet_ramp_steps

        # SEI data
        self.K_Factor = K_Factor
        self.L = L
        self.N = N

        # solid_boundary_data
        self.house_solid_boundary_data = house_solid_boundary_data
        self.ground_solid_boundary_data = ground_solid_boundary_data

        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number,
            mach_number=mach_number,
            characteristic_length_lu=char_length_lu,
            characteristic_length_pu=char_length_pu,
            characteristic_velocity_pu=char_velocity_pu  # reminder: u_char_lu = Ma * cs_lu = Ma * 1/sqrt(3)
        )
        #self.parallel = parallel

        self._solid_mask = np.zeros(shape=self.shape, dtype=bool)  # marks all solid nodes (obstacle, walls, ...)
        #if inlet_bc.casefold() == 'eqin':
        self.in_mask = np.zeros(shape=self.shape, dtype=bool)  # marks all inlet nodes (if EQin is used)
        self.in_mask[0, 1:, :] = True  # inlet in positive x-direction, exklusive "Boden"
        self.ground_mask = np.zeros(shape=self.shape, dtype=bool)  # marks ground on base layer (xy-plane)
        #self.ground_mask[:, 0, :] = True  # mark ground/bottom floor (for standard FWBB/HWBB object-boundary)
        self.house_mask = np.zeros(shape=self.shape, dtype=bool)

        if self.house_solid_boundary_data is not None:
            self.house_mask = self.house_solid_boundary_data.solid_mask
        if self.ground_solid_boundary_data is not None:
            self.ground_mask = self.ground_solid_boundary_data.solid_mask

    @property
    def solid_mask(self):
        if not hasattr(self, '_solid_mask'):
            self.overlap_all_solid_masks()
        return self._solid_mask

    @property
    def non_free_flow_mask(self):
        if not hasattr(self, '_non_free_flow_mask'):
            self.calculate_non_free_flow_mask()
        return self._non_free_flow_mask

    def initial_solution(self, x: torch.Tensor):
        # initial velocity field: "u_init"-parameter [in PU]

        p = np.zeros_like(x[0], dtype=float)[None, ...]
        u = np.zeros((len(x),) + x[0].shape)

        if self.u_init == 0:  # 0: uniform u=0 # "zero"
            print("(INFO) initializing with u_init=zero throughout domain")
            # print(f"u.shape = {u.shape}")
            # print(f"p.shape = {p.shape}")
            pass
        elif self.u_init == 1:  # 1: simple velocity profile everywhere, where there is no other BC  # "profile"
            u[0] = self.wind_speed_profile_power_law(np.where(self.solid_mask, 0, x[1]),
                                           y_ref=self.reference_height_pu,  # REFERENCE height (roof or eg_height)
                                           y_0=self.wsp_y0,
                                           u_ref=self.units.characteristic_velocity_pu, # characteristic velocity at reference height (EG or ROOF)
                                           alpha=self.wsp_alpha)
        elif self.u_init == 2:  # 2: u-profile adjusted to obstacle-geometry  # "profile that is attenuated up to 1/5 of domain length
            # PHILIPPS version with height-shift
            # TODO: implement semi-simple global velocity profile by broadcasting simple WSP-inlet-profile and adjusting height to max. Solid-Height at each XZ-position

            one_fifth_length_index = int(round(self.shape[0]/5))
            k_factor = np.zeros_like(x[0], dtype=float)
            k_factor[0,:,:] = 1
            for x_i in range(one_fifth_length_index):
                k_factor[x_i,:,:] = (one_fifth_length_index-x_i)/one_fifth_length_index

            u[0] = k_factor * self.wind_speed_profile_power_law(np.where(self.solid_mask, 0, x[1]),
                                           y_ref=self.reference_height_pu,  # REFERENCE height (roof or eg_height)
                                           y_0=self.wsp_y0,
                                           u_ref=self.units.characteristic_velocity_pu, # characteristic velocity at reference height (EG or ROOF)
                                           alpha=self.wsp_alpha)


            pass
        else:
            raise NotImplementedError("Specify u_init = 0, 1, or 2")
        
        return p, u

    @property
    def grid(self):
        return makeGrid(self.domain_constraints, self.shape)
        # minx, maxx = self.domain_constraints
        # xyz = tuple(self.units.convert_length_to_pu(torch.linspace(minx[_], maxx[_], self.shape[_]))
        #             for _ in range(self.ndim))  # tuple of lists of x,y,(z)-values/indices
        # return torch.meshgrid(*xyz, indexing='ij')  # meshgrid of x-, y- (und z-)values/indices

    @property
    def boundaries(self):
        print("calling flow.boundaries()")
        time0 = time.time()

        # get grid and masks
        x, y, z = self.grid
        
        # SEI parameters >>>
        L = self.L
        if self.N == 0:
            N = int(np.ceil((self.units.convert_length_to_pu(self.shape[2]) * self.units.convert_length_to_pu(
                self.shape[1])) / (4 * L ** 2)))
        else:
            N = self.N
        # <<<

        # (1/2) overlap solid masks
        self.overlap_all_solid_masks()
        self.calculate_non_free_flow_mask()

        # INLET, OUTLET, TOP/HEAVEN, BOTTOM/GROUND, HOUSE/SOLID
        # ...lateral sides in 3D are periodic BC by default
        # TODO: pass periodicity to Solid BCs (FWBB, HWBB, IBB).

        # initialize wind_speed_profile for inlet BC
        u_inlet_x = self.wind_speed_profile_power_law(np.where(self.solid_mask, 0, y),
                                                      y_ref=self.reference_height_pu,
                                                      # REFERENCE height (roof or eg_height)
                                                      y_0=self.wsp_y0,
                                                      u_ref=self.units.characteristic_velocity_pu,
                                                      # characteristic velocity at reference height (EG or ROOF)
                                                      alpha=self.wsp_alpha)[0, np.newaxis,...]
        u_inlet_y = np.zeros_like(u_inlet_x)
        #print(f"u_inlet_x.shape = {u_inlet_x.shape}")
       # print("u_inlet_x:\n", u_inlet_x)
        u_inlet = np.stack([u_inlet_x, u_inlet_y, u_inlet_y], axis=0)
        #print("u_inlet:\n", u_inlet[0,0,:,0])

        # INLET
        print("initializing inlet boundary condition...")
        if self.inlet_bc.casefold() == 'eqin':
            inlet_boundary_condition = EquilibriumBoundaryPU(self.in_mask, self.units.lattice, self.units, u_inlet)
                                                             # self.wind_speed_profile(np.where(self.solid_mask, 0, x[1]),
                                                             #                         y_ref=self.reference_height_pu, # REFERENCE height (roof_height or eg_height)
                                                             #                         y_0=self.ground_height_pu,
                                                             #                         u_ref=self.units.characteristic_velocity_pu, # characteristic velocity at reference height (EG or ROOF)
                                                             #                         alpha=0.25))
            # inlet_boundary_condition = lt.EquilibriumBoundaryPU(np.abs(x) < 1e-6, self.units.lattice, self.units, u[:, 0, ...], p[0, 0, ...])
        elif self.inlet_bc.casefold() == 'nex':
            inlet_boundary_condition = NonEquilibriumExtrapolationInletU(self.units.lattice, self.units, [-1, 0, 0],
                                                                         np.array(self.initial_solution(self.grid)[1]))  # original aus der Arbeit, Übergibt die y-Koordinaten aller grid-Punkte und erzeugt so die von y abhängige initial solution für das komplette Feld. Wird für MLKs Rechnung in Feld-Größe benötigt
        elif self.inlet_bc.casefold() == 'sei':
            inlet_boundary_condition = SyntheticEddyInlet(self.units.lattice,
                                                          self.units,
                                                          self.grid,  # previously "self.rgrid"
                                                          rho=self.units.convert_density_to_pu(
                                                              self.units.convert_pressure_pu_to_density_lu(0)),
                                                          u_0=self.units.characteristic_velocity_pu,
                                                          K=self.K_Factor * 10,
                                                          L=L, N=N, R=self.reyolds_stress_tensor,
                                                          velocityProfile=self.wind_speed_profile_power_law)
        if self.inlet_bc.casefold() == 'rampeqin':
            inlet_boundary_condition = RampedEquilibriumBoundaryPU(self.in_mask, self.units.lattice, self.units, u_inlet, ramp_steps=self.inlet_ramp_steps)
        else:
            print("(!) flow-class encountered illegal inlet_bc-parameter! Using EquilibriumBoundaryPU")
            inlet_boundary_condition = EquilibriumBoundaryPU(self.in_mask, self.units.lattice, self.units, u_inlet)

        # OUTLET (BACK)
        print("initializing outlet boundary condition...")
        outlet_boundary_condition = None
        if self.outlet_bc.casefold() == 'eqoutp':
            outlet_boundary_condition = EquilibriumOutletP(self.units.lattice, [1, 0, 0], rho0=self.units.convert_pressure_pu_to_density_lu(0))
        elif self.outlet_bc.casefold() == 'eqoutu':
            print("(INFO) Equilibrium Outlet U (!) was selected.")
            out_mask = np.zeros_like(self.solid_mask)
            out_mask[-1, 1:, :] = True
            outlet_boundary_condition = EquilibriumBoundaryPU(out_mask, self.lattice, self.units, u_inlet)
        else:  # default to EQ_outlet_P
            print("(INFO) outlet_bc was not recognized or specified. Defaulting to EQ_outlet_P")
            outlet_boundary_condition = EquilibriumOutletP(self.units.lattice, [1, 0, 0], rho0=self.units.convert_pressure_pu_to_density_lu(0))

        # TOP
        print("initializing top boundary condition...")
        if self.top_bc.casefold() == 'zgo':
            top_boundary_condition = ZeroGradientOutlet(self.units.lattice, [0, 1, 0])
        else:
            top_mask = np.zeros_like(self.solid_mask)
            top_mask[:, -1, :] = True
            top_boundary_condition = EquilibriumBoundaryPU(top_mask, self.units.lattice, self.units, self.initial_solution(self.grid)[0])

        # GROUND
        print("initializing ground boundary condition...")
        ground_boundary_condition = None
        if self.ground_solid_boundary_data is not None:  # ground as IBB
            if self.ground_bc.casefold() == 'ibb':
                ground_boundary_condition = InterpolatedBounceBackBoundary_occ(self.ground_solid_boundary_data.solid_mask,self.lattice, solid_boundary_data=self.ground_solid_boundary_data)
            elif self.ground_bc.casefold() == 'hwbb':
                ground_boundary_condition = HalfwayBounceBackBoundary_occ(self.ground_solid_boundary_data.solid_mask, self.units.lattice, solid_boundary_data=self.ground_solid_boundary_data)
            elif self.ground_bc.casefold() == 'fwbb':
                ground_boundary_condition = FullwayBounceBackBoundary_occ(self.ground_solid_boundary_data.solid_mask, self.units.lattice, solid_boundary_data=self.ground_solid_boundary_data, global_solid_mask=self.solid_mask, periodicity=(False, False, True))
            else:  # default to basic full-mask (fullway) BounceBack
                ground_boundary_condition = BounceBackBoundary(self.ground_solid_boundary_data.solid_mask, self.units.lattice)

        else:
            print(f"(INFO) no ground_solid_boundary_data available to flow.boundaries but ground_bc appears to be wanted. Using self.ground_mask[:, 0, :] = True as ground_mask")
            self.ground_mask[:, 0, :] = True  # TODO: diese Anpassung hier ist irgendwie hässlich, weil ich dann jeweils vorher die global_solid_mask anders habe...
            if self.ground_bc is not None:  # if ground BC is specified for regular SBB
                if self.ground_bc.casefold() == 'hwbb':
                    ground_boundary_condition = HalfwayBounceBackBoundary_occ(self.ground_mask, self.units.lattice, periodicity=(False, False, True))
                elif self.ground_bc.casefold() == 'fwbb':
                    ground_boundary_condition = FullwayBounceBackBoundary_occ(self.ground_mask, self.units.lattice, global_solid_mask=self.solid_mask, periodicity=(False, False, True))
                else:  #default to basic full-mask (fullway) BounceBack
                    ground_boundary_condition = BounceBackBoundary(self.ground_mask, self.units.lattice)


        # HOUSE
        print("initializing house boundary condition...")
        house_boundary_condition = None
        if self.house_solid_boundary_data is not None:
            if self.house_bc.casefold() == 'fwbb':
                house_boundary_condition = FullwayBounceBackBoundary_occ(self.house_solid_boundary_data.solid_mask, self.lattice, self.house_solid_boundary_data, global_solid_mask=self.solid_mask, periodicity=(False, False, True))
            elif self.house_bc.casefold() == 'hwbb':
                house_boundary_condition = HalfwayBounceBackBoundary_occ(self.house_solid_boundary_data.solid_mask, self.lattice, self.house_solid_boundary_data, periodicity=(False, False, True))
                pass
            elif self.house_bc.casefold() == 'ibb' or 'ibb1':
                house_boundary_condition = InterpolatedBounceBackBoundary_occ(self.house_solid_boundary_data.solid_mask, self.lattice, self.house_solid_boundary_data)
            else:
                house_boundary_condition = BounceBackBoundary(self.house_solid_boundary_data.solid_mask, self.lattice)
        else:
            print("(!) flow.boundary(): house_solid_boundary_data is None! no house boundary created!")

        # (2/2) overlap solid masks
        self.overlap_all_solid_masks()
        self.calculate_non_free_flow_mask()

        #>>>
            # ÜBERLEGUNGEN ZUR REIHENFOLGE VON BCs
            # - EQin hat keine NSM, d.h. dort wird durch das streaming regulär der "outlet" rübergeströmt -> PROBLEM
            # - EQ_outP hat NSM -> dort bleibt einfach ein konstanter Wert, des letzten Durchlaufs, im Zweifel also auch die Geschwindigkeit des Nachbarknotens von letztem Step...
            # (!) DENKE: wo kommen Populationen her und wo SOLLTEN sie herkommen?
            # - EQ_out + HWBB -> EQ_out "nimmt" sich Populationen vom Nachbarknoten. welcher diagonal ja ein SOLID Knoten ist! Und dort ist "Null" Geschwindigkeit -> d.h. von dort strömt ETWAS mehr zurück, als erwartbar wäre...
            # - EQ_in + HWBB -> das sollte durch NCM und NSM behebbar sein, weil dann die Pops. des vorherigen Steps einfach "bleiben" / ABER die EQ_in müsste nach hinten, damit "VOR dem Reporter und Streaming die korrekten Pops. stehen"
            #   -> EQ_in hinten: die Pops. werden zwischen Streaming und EQ nochmal kurz von der HWBB angefasst, dann aber wieder überschrieben
            # - vermutlich ist das mit FWBB halt getestet...und damit liefs... weils "IN" der boundary trotzdem "sinnvolle" Fluidpopulationen gibt.
            # (!) aus der Boden-Boundary alle f_index rausnehmen, welche in ihren LU-Koordinaten auf der EQ_in.mask liegen
        #<<<

        # LIST OF BOUNDARIES
        if (ground_boundary_condition is not None) and (house_boundary_condition is not None):
            print("INFO: flow.boundaries contains SEPERATE house and ground solid boundaries")
            boundaries = [
                inlet_boundary_condition,
                outlet_boundary_condition,
                top_boundary_condition,
                ground_boundary_condition,
                house_boundary_condition
            ]
        elif house_boundary_condition is not None:  # if there is no ground_boundary_condition use only one solid_boundary, which is in house_BC
            print("INFO: flow.boundaries contains COMBINED house and ground solid boundaries")
            boundaries = [
                inlet_boundary_condition,
                outlet_boundary_condition,
                top_boundary_condition,
                house_boundary_condition
            ]
        elif ground_boundary_condition is not None:  # if there is no house_boundary_condition use only one solid_boundary, which is in ground_BC
            print("INFO: flow.boundaries contains ONLY ground solid boundary")
            boundaries = [
                inlet_boundary_condition,
                outlet_boundary_condition,
                top_boundary_condition,
                ground_boundary_condition
            ]
        else:
            print("(!) No solid boundary condition given... will produce only bullshit")
            boundaries = [
                inlet_boundary_condition,
                outlet_boundary_condition,
                top_boundary_condition
            ]

        i = 0
        for boundary in boundaries:
            print(f"boundaries[{i}]: {str(boundary)}")
            i += 1

        # exclude solid nodes (and non-free-flow nodes) from f_indices of all solid boundaries
        print("exlcuding solid nodes from f_index(_gt_lt) of bounce back boundaries")
        for boundary in boundaries:
            if hasattr(boundary, 'f_index'):
                num_entries = boundary.f_index.shape[0]
                print(f"boundary {boundary} has f_index with {num_entries} entries")
                if boundary.f_index.shape[0] > 0:
                    boundary.f_index = boundary.f_index[torch.where(~self.lattice.convert_to_tensor(self.non_free_flow_mask)[boundary.f_index[:, 1], boundary.f_index[:, 2], boundary.f_index[:, 3] if len(self.shape) == 3 else None])]
                print(f"removed {num_entries - boundary.f_index.shape[0]} entries")
            if hasattr(boundary, 'f_index_fwbb'):
                num_entries = boundary.f_index_fwbb.shape[0]
                print(f"boundary {boundary} has f_index_fwbb with {num_entries} entries, but is FWBB so no cleanup possible. Provide global_solid_mask to FWBB-initialization, if you need cleanup of f_index_fwbb")
                # if boundary.f_index_fwbb.shape[0] > 0:
                #     boundary.f_index_fwbb = boundary.f_index_fwbb[torch.where(~self.lattice.convert_to_tensor(np.logical_and(self.solid_mask, ~boundary.mask))[boundary.f_index_fwbb[:, 1], boundary.f_index_fwbb[:, 2], boundary.f_index_fwbb[:, 3] if len(self.shape) == 3 else None])]
                # print(f"removed {num_entries - boundary.f_index_fwbb.shape[0]} entries")
            if hasattr(boundary, 'f_index_gt'):
                num_entries = boundary.f_index_gt.shape[0]
                print(f"boundary {boundary} has f_index_gt with {num_entries} entries")
                if boundary.f_index_gt.shape[0] > 0:
                    boundary.d_gt = boundary.d_gt[torch.where(~self.lattice.convert_to_tensor(self.non_free_flow_mask)[boundary.f_index_gt[:, 1], boundary.f_index_gt[:, 2], boundary.f_index_gt[:, 3] if len(self.shape) == 3 else None])]
                    boundary.f_index_gt = boundary.f_index_gt[torch.where(~self.lattice.convert_to_tensor(self.non_free_flow_mask)[boundary.f_index_gt[:, 1], boundary.f_index_gt[:, 2], boundary.f_index_gt[:, 3] if len(self.shape) == 3 else None])]
                print(f"removed {num_entries - boundary.f_index_gt.shape[0]} entries")
            if hasattr(boundary, 'f_index_lt'):
                num_entries = boundary.f_index_lt.shape[0]
                print(f"boundary {boundary} has f_index_lt with {num_entries} entries")
                if boundary.f_index_lt.shape[0] > 0:
                    boundary.d_lt = boundary.d_lt[torch.where(~self.lattice.convert_to_tensor(self.non_free_flow_mask)[boundary.f_index_lt[:, 1], boundary.f_index_lt[:, 2], boundary.f_index_lt[:, 3] if len(self.shape) == 3 else None])]
                    boundary.f_index_lt = boundary.f_index_lt[torch.where(~self.lattice.convert_to_tensor(self.non_free_flow_mask)[boundary.f_index_lt[:, 1], boundary.f_index_lt[:, 2], boundary.f_index_lt[:, 3] if len(self.shape) == 3 else None])]
                print(f"removed {num_entries - boundary.f_index_lt.shape[0]} entries")

        # time execution of flow.boundary()
        time1 = time.time() - time0
        print(f"boundaries took {floor(time1 / 60):02d}:{floor(time1 % 60):02d} [mm:ss].")
        return boundaries

    def overlap_all_solid_masks(self):
        print("overlap_all_solid_masks")
        time0 = time.time()
        ###assert self.boundary_objects is not None
            # boundaries_list = [_ for _ in self.boundary_objects
            #                    if _.unique_boundary and _.boundary_type is not PartiallySaturatedBoundary]
        #boundaries_list = self.boundaries
        # for boundary in boundaries_list:
        #     if not hasattr(boundary.collision_data, 'solid_mask'):
        #         boundary.calculate_points_inside()
        # COMBINE ALL SOLID-Masks to single mask
        # self._solid_mask = torch.zeros_like(boundaries_list[0].solid_mask, dtype=torch.bool, device=self.lattice.device)
        # for boundary in boundaries_list:
        #     self._solid_mask = self.solid_mask | boundary.solid_mask.to(self.lattice.device)

        # TODO: geht das irgendwie anders? ich kann nicht self.boundaries aufrufen, weil die boundaries dann komplett neu initialisiert werden und das Zeit kostet!
        self._solid_mask = np.zeros(shape=self.shape, dtype=bool)
        self._solid_mask = self.solid_mask | self.house_mask | self.ground_mask
        # TODO: falls hier weitere solids hinzugefügt werden (in diesem flow), dann müssen deren Masken nach Erstellung noch entsprechend verschnitten werden...
        #  alternativ: man könnte ein "update solid mask" oderso machen, in dem man dann alle True Punkte hinzufügt... und das wird von einer boundary selbst bei initialisierung aufgerufen
        time1 = time.time() - time0
        print(f"overlap_all_solid_masks took {floor(time1 / 60):02d}:{floor(time1 % 60):02d} [mm:ss].")
        return

    def calculate_non_free_flow_mask(self):
        print("calculating non_free_flow_mask")
        time0 = time.time()
        self._non_free_flow_mask = np.zeros(shape=self.shape, dtype=bool)

        # (!) outlet is currently defined through "direction" and not mask, this is why this is "ghetto-style" implemented locally like this
        out_mask = np.zeros_like(self.solid_mask)
        out_mask[-1, 1:, :] = True

        self._non_free_flow_mask = self.solid_mask | self.house_mask | self.ground_mask | out_mask | self.in_mask
        time1 = time.time() - time0
        print(f"calculate_non_free_flow_mask took {floor(time1/ 60):02d}:{floor(time1 % 60):02d} [mm:ss].")
        return

    def wind_speed_profile(self, y, y_ref=None, y_0=None, u_ref=None, alpha=0.25):
        # exponential wind speed profile from QUELLE, with ZERO velocity at y_0 and u_ref at y_ref+y_0
        # alle Angaben in PU, y in absolute PU relative to (0,0,0)
        # FRAGE: soll y_ref die Höhe vom Boden oder die Absoluthöhe sein?
            # - erstmal vom Boden, d.h. ich muss da y_0 noch draufrechnen, für absolute PU-Koordinaten, welche ja in y übergeben werden!
        if y_ref == None:
            y_ref = self.reference_height_pu

        if y_0 == None:
            y_0 = self.ground_height_pu

        # TODO: y_0 is the ground_height and reference ZERO-height for profile. add u_0_height at which the profile starts and is stretched, u_ref at y_ref stays relative to y_0
        #return torch.where(y < y_0, 0, u_ref * ((y - y_0) / y_ref) ** alpha)
        #print("y_0, y_ref, alpha:", y_0, y_ref, alpha)
        # print("y:", y)
       # print("WSP is:\n", np.where(y <= y_0, 0, u_ref * ((y - y_0) / y_ref) ** alpha))
        return np.where(y <= y_0, 0, u_ref * (np.where(y <= y_0, 0, (y - y_0)) / y_ref) ** alpha)

    def wind_speed_profile_power_law(self, y, y_ref=None, y_0=None, u_ref=None, alpha=0.25):
        # INPUTS REF: z_0, z_min, z_max, c_0, v_b, kat.,
        # WIND SPEED Power law: u(height)/u_ref = (height/height_ref)^alpha
        # known reference speed u_ref at height y_ref. Extrapolates wind speed u at y (in m)
        # alpha: empirically derived coefficient, depends on atmospheric stability (wikipedia 1/7~1.43; Tominaga/MLK: 0.25)

        # REF: DIN EN 1991-1-4 NA.B (Nationaler Anhang, gültig für Deutschland. Abseits von DE ist auf das log-Profil aus DIN EN 1991-1-4 zu verweisen!
        #
        # Mittlere Windgeschwindigkeit v_m(z) = c_f(z) * c_0(z) * v_b
        #   c_r Rauigkeitsfaktor/-beiwert abh. von Geländekategorien (s.u.)
        #   c_0 Topographiebeiwert (bei Höhenunterschieden vor/hinter Gebäude, die die mittlere Windgeschwindigkeit um min. 5% erhöhen)
        #   v_b Basisgeschwindigkeit in m/s = Mittelwert über 10 min in 10 m Höhe (0.02% Überschreiwuntswahrscheinlichkeit).
        #           - v_b0 unabhängig von Windrichtugn und Jahreszeit -> v_b = c_dir * c_season * v_b,0 /beides aber für DE 1.0 (!)
        #           - 10m über Grund bei ebenem, offenen Gelände mit niedriger Vegetation (gras oder Hindernisse die 20mal so weit auseinander wie hoch sind)
        #           -  Entspricht Gelände-Kat. II, z.B. auf Flughäfen
        #           Windzonen: 1: 22.5 m/s, Kat. 2: 25 m/s, Kat. 3: 27.5 m/s, Kat. 4: 30 m/s
        #   Bei Einfluss von Höhe (>800 ü NN) siehe Quelle LASTANNAHMEN Kap. 6.8 Widngeschwindigkei und Geschwindigkeitsdruck...
        # * darf bis z=300m angenommen werden
        #
        # Rauigkeitsbeiwert c_r(z) = v_m(z)/v_m(10) = 0.19*(z_0/0.05)^0.07 * ln(10/z_0) * (z/10)^alpha
        #   alpha   Profilexponent abh. von Geländekat. I: 0.12, II: 0.16, III: 0.22, IV: 0.3
        #   z_0     Rauigkeitslänge:                    I: 0.01. II: 0.05, III: 0.30, IV: 1.05
        #   z       Höhe über Grund
        #   z_min   Mindesthöhe:                        I: 2m,   II: 4m,   III: 8m,   IV: 16m
        #   (!) laut DIN EN... ist c_r(z) für z<z_min konstant, d.h. unterhalb einer Mindeshöhe ist gleichförmige Rauigkeit über die Höhe anzunehmen, d.h. es gibt einen "Sockelbetrag" der Geschwindigkeit am Boden
        #       => also c_r(z<z_min) = const. = c_r(z_min)
        #          (!) z_min wird zwar in der Formel 6.17 nicht explizit erwähnt, in folgenden Bespielen aber als Sockelbetrag wie im LOG-profil verwendet...

        # entsprechend ergeben sich für die Kat. I-IV in DE:
        #   I: v_m=1.18*v_b*(z/10)^0.12
        #  II: v_m=1.00*v_b*(z/10)^0.16
        # III: v_m=0.77*v_b*(z/10)^0.22  (DRUCKFEHLER IN Tab. 6.16 anzunehmen! für die alpha-Werte)
        #  IV: v_m=0.56*v_b*(z/10)^0.3
        # für vm(z<z_min) sind es: I: 0.97*v_b, II: 0.86*v_b, III: 0.73*v_b, IV: 0.64*v_b
        # (Mischgelände siehe Tab- 6.16)
        #   v_b Basisgeschwindigkeit in Abh. der Windzone in m/s:
        #       Windzone 1: 22.5 m/s, Kat. 2: 25 m/s, Kat. 3: 27.5 m/s, Kat. 4: 30 m/s


        # (!) all units in PU
        # y is absolute PU-coordinates relative to grid

        ### c_r_min = 0.19*(y_0/0.05)^0.07 * np.log(10/y_0) * (y_min/10)^alpha
        ### c_r = 0.19*(y_0/0.05)^0.07 * np.log(10/y_0) * (y/10)^alpha       # SOLLTE übereinstimmen mit: v_m... oben für DE
        # TODO: plot Kat 1, Kat2, Kat3 against MLK Parameters...
        #       for Kat-Formulas, use formula and condensed formula respectively and compare.

        if self.wsp_shift_up_pu != 0:
            print(f"(WSP_powerLaw): wsp_shift_up_pu is not 0! WSP is shifted up by {self.wsp_shift_up_pu} meters!")
            # (!) WSP_SHIFT only works, if y_0 is not set...

        if u_ref is None:
            print(f"(WSP_powerLaw): u_ref for WSP_powerLaw is not set, taking U_char as u-ref")
            u_ref = self.units.characteristic_velocity_pu

        if y_ref == None:
            # absolute reference height in PU-grid-coordinates; height at which u_ref is present
            y_ref = self.reference_height_pu+self.ground_height_pu+self.wsp_shift_up_pu

        if y_0 == None:
            # absolute zero-height of WSP in PU-grid-coordinates; height at and below which u(y<=y_0)=0
            y_0 = self.ground_height_pu+self.wsp_shift_up_pu

        # Q: reference y_0 to PU=LU=0 or to ground_height? -> reference to PU=0
        # Q: reference y_ref to PU=LU=0 or to ground height, or to y_0? -> reference to PU=0

        u_profile = np.where(y <= y_0, 0, u_ref * (np.where(y <= y_0, 0, (y - y_0)) / (y_ref-y_0)) ** alpha)
        # if len(u_profile.shape) == 1:
        #     u_profile_deltas = u_profile[1:] - u_profile[:-1]
        # elif len(u_profile.shape) == 2:
        #     u_profile_deltas = u_profile[0, :][1:] - u_profile[0, :][:-1]
        # elif len(u_profile.shape) == 3:
        #     u_profile_deltas = u_profile[0, :, 0][1:] - u_profile[0, :, 0][:-1]

        #print("(!) (WSP_powerLaw): max. dux/dy gradient in inflow-profile is:", max(abs(u_profile_deltas)))
        # POWER LAW: from u(y<=y_0)=0 to u(y=y_ref)=u_ref with exponent alpha
        return u_profile #np.where(y <= y_0, 0, u_ref * (np.where(y <= y_0, 0, (y - y_0)) / (y_ref-y_0)) ** alpha)


    def wind_speed_profile_turb(self, y, u_0):
        ## entspricht 3D_literature_neue_Boundary "wsp" und TestSEMBoundary/Empty/...
        ## hält den Minimalwert auf z_min fest und lässt ihn nicht für z=0 auf 0 abfallen!
        # based on DIN Onstwaswindprofil
        # all inputs in PU
        # return 0.77 * u_max * (z / 10) ** (alpha)
        roughness_length = y_0 = 0.02  # meter
        k_r = 0.19 * (y_0 / 0.05) ** 0.07
        y_min = 1.2

        ##(MK_torch-Version): return torch.where(y < y_min, u_0 * k_r * torch.log(torch.tensor(y_min / y_0, device=self.units.lattice.device)) * 1, u_0 * k_r * torch.log(y / y_0) * 1)
        return np.where(y < y_min,
                           u_0 * k_r * np.log(y_min / y_0) * 1,
                           u_0 * k_r * np.log(y / y_0) * 1)

    def wind_speed_profile_log_law(self, y, u_0, y_min=1.2, y_0=0.02, kat=2):
        # INPUTS Literatur: v_b, c_0=1, z_0, z_min, z_max, Kat.
        # INPUTS hier:      u_0, -    , y_0, y_min, -    , kat

        ## entspricht 3D_literature_neue_Boundary "wsp" und TestSEMBoundary/Empty/...
        ## hält den Minimalwert auf z_min fest und lässt ihn nicht für z=0 auf 0 abfallen!

        # REF DIN 1991-1-4 nach QUELLE lastannahmen, QQuelle WIndlasten (Kap.6)
        # Mittlere Windgeschwindigkeit v_m(z) = c_f(z) * c_0(z) * v_b
        #   c_r Rauigkeitsfaktor/-beiwert abh. von Geländekategorien
        #   c_0 Topographiebeiwert (bei Höhenunterschieden vor/hinter Gebäude, die die mittlere Windgeschwindigkeit um min. 5% erhöhen)
        #   v_b Basisgeschwindigkeit in m/s = Mittelwert über 10 min in 10 m Höhe (0.02% Überschreiwuntswahrscheinlichkeit).
        #           - v_b0 unabhängig von Windrichtugn und Jahreszeit -> v_b = c_dir * c_season * v_b,0 /beides aber für DE 1.0 (!)
        #           - 10m über Grund bei ebenem, offenen Gelände mit niedriger Vegetation (gras oder Hindernisse die 20mal so weit auseinander wie hoch sind)
        #           -  Entspricht Gelände-Kat. II, z.B. auf Flughäfen
        #           v_b nach Windzonen: 1: 22.5 m/s, Kat. 2: 25 m/s, Kat. 3: 27.5 m/s, Kat. 4: 30 m/s

        # Rauigkeitsbeiwert c_r(z) = k_r*ln(z/z_0)  für  z_min < z < z_max
        #                   c_r(z) = c_r(z_min)     für  z < z_min
        #   z_0 Rauigkeitslänge (m):  (siehe auch unten z_0,Kat)
        #   z Bezugshöhe (m)
        #   z_min Mindesthöhe (m):    0: 1.0,    I: 1.0,  II: 2.0,  III: 5.0, IV: 10.0
        #   z_max = 200 m (darüber verleirt das Profil seine Gültigkeit)
        #   k_r Rauigkeitsfaktor abh. von Rauigkeitslänge nach:
        #       k_r = 0.19*(z_0/z_0,KAT)^0.07  für eine bestimmte z_0 einer KAT.
        #       z_0, KAT Rauigkeitslänge (m):  0: 0.0003, I: 0.01, II: 0.05, III: 0.3, IV: 1.0
        #       LAUT Tab. 6.11 in Quelle Windlasten:
        #           kr: 0: 0.1560, I: 0.1698, II: 0,1900, III: 0.2154, IV: 0.2343

        # all inputs in PU

        # Zuordnung Quelle, "meine Parameter"
        #           z = y
        #         v_b = u_0

        # MLK Parameter:
        y_0 = 0.02  # meter, roughness_length
        y_min = 1.2
        y_max = 200
        z_0_kat2 = 0.05
        k_r = 0.19 * (y_0 / z_0_kat2) ** 0.07  # hier k_r ~ 0.1782, das ist zw. Kapt. I und II // 0.05 ist z_0,II für Kat.2

        ##(MK_torch-Version): return torch.where(y < y_min, u_0 * k_r * torch.log(torch.tensor(y_min / y_0, device=self.units.lattice.device)) * 1, u_0 * k_r * torch.log(y / y_0) * 1)
        return np.where(y < y_min,
                           u_0 * k_r * np.log(y_min / y_0),  # const. unterhalb der Mindeshöhe
                           u_0 * k_r * np.log(y / y_0)       # log-Profil oberhalb der Mindeshöhe (streng genommen bis zur max-Höhe)
                        )

    def reyolds_stress_tensor(self, z, u_0):
        # z is y is height, depending on your coordinate system
        """
        inputs in PU
        [I_x² * U_x_Mean², ~0, experiment]
        [~0, I_y² * U_y_Mean², ~0]
        [experiment , ~0, 9I_w² * U_z_Mean²]
        """
        house_length_pu = 10
        #roof_height_pu = self.units.convert_length_to_pu(self.eg_height)  # house_length_pu * 1.25
        roof_height_pu = self.reference_height_pu
        stress = torch.ones(z.shape + (3, 3), device=self.units.lattice.device)
        z_0 = 0.02
        z_min = 1.2
        #TODO: which WSP to take: originally this was done with "wind speed profile" -> what does MLK take?
        stress[..., 0, 0] = torch.where(z > z_min, (
                    (1 - 2e-4 * (np.log10(z_0) + 3) ** 6) / torch.log(z / z_0)) ** 2 * self.wind_speed_profile(z,
                                                                                                               u_0) ** 2,
                                        ((1 - 2e-4 * (np.log10(z_0) + 3) ** 6) / np.log(
                                            z_min / z_0)) ** 2 * self.wind_speed_profile(
                                            torch.tensor(z_min, device=self.units.lattice.device), u_0) ** 2)
        stress[..., 0, 1] = stress[..., 1, 0] = 0
        stress[..., 0, 2] = stress[..., 2, 0] = (0.4243 * (z / 100) ** 2 - 2.288 * (
                    z / 100) - 2) * 1e-3 * u_0 ** 2  # function fitted to data from experiment
        stress[..., 1, 1] = (0.88 / torch.log(
            (z + 0.00001) * (0.33 / roof_height_pu) * 1e5 / 2.5)) ** 2 * self.wind_speed_profile(z, u_0) ** 2
        stress[..., 1, 2] = stress[..., 2, 1] = 0
        stress[..., 2, 2] = 0.08 ** 2 * self.wind_speed_profile(z, u_0) ** 2

        return stress * self.units.convert_density_to_pu(self.units.convert_pressure_pu_to_density_lu(0))

