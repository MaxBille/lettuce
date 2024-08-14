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
    HalfwayBounceBackBoundary_compact_v3, PartiallySaturatedBoundary
from lettuce.boundary_mk import NonEquilibriumExtrapolationInletU, SyntheticEddyInlet, ZeroGradientOutlet
from pspelt.obstacleFunctions import makeGrid
import time


# houseFlow3D by M.Kliemank, from MA-Thesis-CD-ROM "simulation_code.py"
class HouseFlow3D(object):
    shape: tuple[int, int, int] or tuple[int, int]

    def __init__(self, shape: tuple, reynolds_number: float, mach_number: float, lattice: Lattice,
                 domain_constraints: tuple, char_length_lu: float,
                 char_length_pu: float = 1, char_velocity_pu=1, u_init: 0 or 1 or 2 = 0,
                 reference_height_pu = 0, ground_height_pu = 0,
                 inlet_bc: str = "eqin", outlet_bc: str = "eqoutp", ground_bc: str = "fwbb", house_bc: str = "fwbb",
                 top_bc: str = "zgo",
                 house_solid_boundary_data = None,
                 ground_solid_boundary_data = None,
                 K_Factor=10,  # K_factor for SEI boundary inlet
                 L=3,  # L for SEI
                 N=34,  # N number of random voctices for SEI
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

        # bc_types
        self.inlet_bc = inlet_bc
        self.outlet_bc = outlet_bc
        self.ground_bc = ground_bc
        self.house_bc = house_bc
        self.top_bc = top_bc

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
        if inlet_bc.casefold() == 'eqin':
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

    def initial_solution(self, x: torch.Tensor):
        # initial velocity field: "u_init"-parameter

        p = np.zeros_like(x[0], dtype=float)[None, ...]
        u = np.zeros((len(x),) + x[0].shape)

        if self.u_init == 0:  # 0: uniform u=0
            print("(INFO) initializing with u_init=zero throughout domain")
            # print(f"u.shape = {u.shape}")
            # print(f"p.shape = {p.shape}")
            pass
        elif self.u_init == 1:  # 1: simple velocity profile everywhere, where there is no other BC
            u[0] = self.wind_speed_profile(np.where(self.solid_mask, 0, x[1]),
                                           y_ref=self.reference_height_pu,  # REFERENCE height (roof or eg_height)
                                           y_0=self.ground_height_pu,
                                           u_ref=self.units.characteristic_velocity_pu,
                                           # characteristic velocity at reference height (EG or ROOF)
                                           alpha=0.25)
            # self.wind_speed_profile(np.where(self.solid_mask, 0, y)[0],
            #                         y_ref=self.reference_height_pu,  # REFERENCE height (roof_height or eg_height)
            #                         y_0=self.ground_height_pu,
            #                         u_ref=self.units.characteristic_velocity_pu,
            #                         # characteristic velocity at reference height (EG or ROOF)
            #                         alpha=0.25)
            # TODO: implement simple global velocity profile by broadcasting simple WSP-inlet-profile
        elif self.u_init == 2:  # 2: u-profile adjusted to obstacle-geometry
            # PHILIPPS version with height-shift
            # TODO: implement semi-simple global velocity profile by broadcasting simple WSP-inlet-profile and adjusting height to max. Solid-Height at each XZ-position
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

        # INLET, OUTLET, TOP/HEAVEN, BOTTOM/GROUND, HOUSE/SOLID
        # ...lateral sides in 3D are periodic BC by default
        # TODO: pass periodicity to Solid BCs (FWBB, HWBB, IBB).

        # initialize wind_speed_profile for inlet BC
        u_inlet_x = self.wind_speed_profile(np.where(self.solid_mask, 0, y),
                                     y_ref=self.reference_height_pu, # REFERENCE height (roof_height or eg_height)
                                     y_0=self.ground_height_pu,
                                     u_ref=self.units.characteristic_velocity_pu, # characteristic velocity at reference height (EG or ROOF)
                                     alpha=0.25)[0, np.newaxis,...]
        u_inlet_y = np.zeros_like(u_inlet_x)
        print(f"u_inlet_x.shape = {u_inlet_x.shape}")
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
                                                                         np.array(self.initial_solution(self.grid)[1]))  # original aus der Arbeit
        elif self.inlet_bc.casefold() == 'sei':
            inlet_boundary_condition = SyntheticEddyInlet(self.units.lattice,
                                                          self.units,
                                                          self.grid,  # previously "self.rgrid"
                                                          rho=self.units.convert_density_to_pu(
                                                              self.units.convert_pressure_pu_to_density_lu(0)),
                                                          u_0=self.units.characteristic_velocity_pu,
                                                          K=self.K_Factor * 10,
                                                          L=L, N=N, R=self.reyolds_stress_tensor,
                                                          velocityProfile=self.wind_speed_profile)
        else:
            print("(!) flow-class encountered illegal inlet_bc-parameter! Using EquilibriumBoundaryPU")
            inlet_boundary_condition = EquilibriumBoundaryPU(self.in_mask, self.units.lattice, self.units, u_inlet)

        # OUTLET (BACK)
        print("initializing outlet boundary condition...")
        out_mask = np.zeros_like(self.solid_mask)
        out_mask[-1, 1:, :] = True  # not needed for directional boundaries...
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
        if self.house_bc.casefold() == 'fwbb':
            house_boundary_condition = FullwayBounceBackBoundary_occ(self.house_solid_boundary_data.solid_mask, self.lattice, self.house_solid_boundary_data, global_solid_mask=self.solid_mask, periodicity=(False, False, True))
        elif self.house_bc.casefold() == 'hwbb':
            house_boundary_condition = HalfwayBounceBackBoundary_occ(self.house_solid_boundary_data.solid_mask, self.lattice, self.house_solid_boundary_data, periodicity=(False, False, True))
            pass
        elif self.house_bc.casefold() == 'ibb' or 'ibb1':
            house_boundary_condition = InterpolatedBounceBackBoundary_occ(self.house_solid_boundary_data.solid_mask, self.lattice, self.house_solid_boundary_data)
        else:
            house_boundary_condition = BounceBackBoundary(self.house_solid_boundary_data.solid_mask, self.lattice)

        # (1/2) overlap solid masks
        self.overlap_all_solid_masks()

        if ground_boundary_condition is not None:
            print("INFO: flow.boundaries contains SEPERATE house and ground solid boundaries")
            boundaries = [
                inlet_boundary_condition,
                outlet_boundary_condition,
                top_boundary_condition,
                ground_boundary_condition,
                house_boundary_condition
            ]
            i = 0
            for boundary in boundaries:
                print(f"boundaries[{i}]: {str(boundary)}")
                i += 1
        else:  # if there is no ground_boundary_condition use only one solid_boundary, which is in house_BC
            print("INFO: flow.boundaries contains COMBINED house and ground solid boundaries")
            boundaries = [
                inlet_boundary_condition,
                outlet_boundary_condition,
                top_boundary_condition,
                house_boundary_condition
            ]
            i = 0
            for boundary in boundaries:
                print(f"boundaries[{i}]: {str(boundary)}")
                i += 1
        # exclude solid nodes from f_indices of all solid boundaries
        print("exlcuding solid nodes from f_index(_gt_lt) of bounce back boundaries")
        for boundary in boundaries:
            if hasattr(boundary, 'f_index'):
                num_entries = boundary.f_index.shape[0]
                print(f"boundary {boundary} has f_index with {num_entries} entries")
                if boundary.f_index.shape[0] > 0:
                    boundary.f_index = boundary.f_index[torch.where(~self.lattice.convert_to_tensor(self.solid_mask)[boundary.f_index[:, 1], boundary.f_index[:, 2], boundary.f_index[:, 3] if len(self.shape) == 3 else None])]
                print(f"removed {num_entries - boundary.f_index.shape[0]} entries")
            if hasattr(boundary, 'f_index_fwbb'):
                num_entries = boundary.f_index_fwbb.shape[0]
                print(f"boundary {boundary} has f_index with {num_entries} entries, but is FWBB so no cleanup possible. Provide global_solid_mask to FWBB-initialization, if you need cleanup of f_index_fwbb")
                # if boundary.f_index_fwbb.shape[0] > 0:
                #     boundary.f_index_fwbb = boundary.f_index_fwbb[torch.where(~self.lattice.convert_to_tensor(np.logical_and(self.solid_mask, ~boundary.mask))[boundary.f_index_fwbb[:, 1], boundary.f_index_fwbb[:, 2], boundary.f_index_fwbb[:, 3] if len(self.shape) == 3 else None])]
                # print(f"removed {num_entries - boundary.f_index_fwbb.shape[0]} entries")
            if hasattr(boundary, 'f_index_gt'):
                num_entries = boundary.f_index_gt.shape[0]
                print(f"boundary {boundary} has f_index_gt with {num_entries} entries")
                if boundary.f_index_gt.shape[0] > 0:
                    boundary.d_gt = boundary.d_gt[torch.where(~self.lattice.convert_to_tensor(self.solid_mask)[boundary.f_index_gt[:, 1], boundary.f_index_gt[:, 2], boundary.f_index_gt[:, 3] if len(self.shape) == 3 else None])]
                    boundary.f_index_gt = boundary.f_index_gt[torch.where(~self.lattice.convert_to_tensor(self.solid_mask)[boundary.f_index_gt[:, 1], boundary.f_index_gt[:, 2], boundary.f_index_gt[:, 3] if len(self.shape) == 3 else None])]
                print(f"removed {num_entries - boundary.f_index_gt.shape[0]} entries")
            if hasattr(boundary, 'f_index_lt'):
                num_entries = boundary.f_index_lt.shape[0]
                print(f"boundary {boundary} has f_index_lt with {num_entries} entries")
                if boundary.f_index_lt.shape[0] > 0:
                    boundary.d_lt = boundary.d_lt[torch.where(~self.lattice.convert_to_tensor(self.solid_mask)[boundary.f_index_lt[:, 1], boundary.f_index_lt[:, 2], boundary.f_index_lt[:, 3] if len(self.shape) == 3 else None])]
                    boundary.f_index_lt = boundary.f_index_lt[torch.where(~self.lattice.convert_to_tensor(self.solid_mask)[boundary.f_index_lt[:, 1], boundary.f_index_lt[:, 2], boundary.f_index_lt[:, 3] if len(self.shape) == 3 else None])]
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

    def wind_speed_profile(self, y, y_ref=None, y_0=0, u_ref=None, alpha=0.25):
        # exponential wind speed profile from QUELLE, with ZERO velocity at y_0 and u_ref at y_ref+y_0
        # alle Angaben in PU, y in absolute PU relative to (0,0,0)
        # FRAGE: soll y_ref die Höhe vom Boden oder die Absoluthöhe sein?
            # - erstmal vom Boden, d.h. ich muss da y_0 noch draufrechnen, für absolute PU-Koordinaten, welche ja in y übergeben werden!
        if y_ref == None:
            y_ref = self.reference_height_pu

        #return torch.where(y < y_0, 0, u_ref * ((y - y_0) / y_ref) ** alpha)
        print("y_0, y_ref, alpha:", y_0, y_ref, alpha)
        # print("y:", y)
        # TODO: fix runtime warning, that (y-y_0)**alpha produces error with fractional power of negative number, even though the specific calc. doesn't have to be done, because y<=y_0
       # print("WSP is:\n", np.where(y <= y_0, 0, u_ref * ((y - y_0) / y_ref) ** alpha))
        return np.where(y <= y_0, 0, u_ref * (np.where(y <= y_0, 0, (y - y_0)) / y_ref) ** alpha)

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