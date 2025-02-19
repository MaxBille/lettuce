from math import ceil, floor, sqrt
import numpy as np
import torch
from lettuce.lattices import Lattice  # for TYPING purposes
from lettuce.unit import UnitConversion
from lettuce.util import append_axes
from lettuce.boundary import EquilibriumBoundaryPU, \
    BounceBackBoundary, HalfwayBounceBackBoundary, FullwayBounceBackBoundary, EquilibriumOutletP, AntiBounceBackOutlet, \
    InterpolatedBounceBackBoundary, InterpolatedBounceBackBoundary_compact_v1, \
    InterpolatedBounceBackBoundary_compact_v2, InterpolatedBounceBackBoundary_occ, \
    SlipBoundary, FullwayBounceBackBoundary_compact, FullwayBounceBackBoundary_occ, \
    HalfwayBounceBackBoundary_compact_v1, HalfwayBounceBackBoundary_compact_v2, HalfwayBounceBackBoundary_occ, \
    HalfwayBounceBackBoundary_compact_v3, PartiallySaturatedBoundary, RampedEquilibriumBoundaryPU
from lettuce.boundary_mk import NonEquilibriumExtrapolationInletU, SyntheticEddyInlet, ZeroGradientOutlet
from pspelt.obstacleFunctions import makeGrid
import time

# "Keil" Flow, to test response of flow field and boundary conditions to velocity gradients in the inflow or field


class VelocityTestFlow:

    def __init__(self, shape: tuple, reynolds_number: float, mach_number: float, lattice: Lattice,
                 domain_constraints: tuple, char_length_lu: float,
                 char_length_pu: float = 1, char_velocity_pu = 1, u_init: int = 0,
                 char_density_pu=1,
                 inlet_y_rel_start = None, inlet_y_rel_end = None,
                 inlet_velocity_pu = None,  # u_PU/x_LU
                 inlet_bc: str = "equin",
                 inlet_ramp_steps = 0,
                 outlet_bc: str = "eqoutp",
                 lateral_bc: str = "periodic",
                 ):

        self.shape = shape
        self.lattice = lattice

        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number,
            mach_number=mach_number,
            characteristic_length_lu=char_length_lu,
            characteristic_length_pu=char_length_pu,
            characteristic_velocity_pu=char_velocity_pu,  # reminder: u_char_lu = Ma * cs_lu = Ma * 1/sqrt(3)
            characteristic_density_pu=char_density_pu
        )

        self.domain_constraints = domain_constraints
        self.u_init = u_init
        self.inlet_y_rel_start = inlet_y_rel_start
        self.inlet_y_rel_end = inlet_y_rel_end

        if inlet_velocity_pu is not None:
            self.inlet_velocity_pu = inlet_velocity_pu
        else:
            print("FLOW: no inlet_velocity_pu specified. setting inlet_velocity_pu to char_velocity_pu...")
            self.inlet_velocity_pu = char_velocity_pu

        self.inlet_bc = inlet_bc
        self.outlet_bc = outlet_bc
        self.lateral_bc = lateral_bc

        self.in_mask = np.zeros(shape=self.shape, dtype=bool)
        self.in_mask[0,:,:] = True  # inlet on the left
        self.solid_mask = np.zeros(shape=self.shape, dtype=bool)

        self.inlet_ramp_steps = inlet_ramp_steps

        self.u_x_profile_3D_values = self.u_x_profile_3D(self.grid[1], inlet_velocity_pu = self.inlet_velocity_pu)

    def initial_solution(self, x: torch.Tensor):
        p = np.zeros_like(x[0], dtype=float)[None, ...]
        u = np.zeros((len(x),) + x[0].shape)

        if self.u_init == 0:
            # 0: uniform u=0 # "zero"
            print("(INFO) initializing with u_init=zero throughout domain")
            # print(f"u.shape = {u.shape}")
            # print(f"p.shape = {p.shape}")
            pass
        elif self.u_init == 1:  # 1: simple velocity profile everywhere, where there is no other BC  # "profile"
            #TODO: initialize whole domain with profile
            u[0] = self.u_x_profile_3D_values # self.keil_profil(x[1], keil_percentage_of_inlet=self.keil_percentage_of_inlet, keil_u_max=self.units.characteristic_velocity_pu)
            pass
        elif self.u_init == 2:
            one_fifth_length_index = int(round(self.shape[0] / 5))
            k_factor = np.zeros_like(x[0], dtype=float)
            k_factor[0, :, :] = 1
            for x_i in range(one_fifth_length_index):
                k_factor[x_i, :, :] = (one_fifth_length_index - x_i) / one_fifth_length_index

            u[0] = k_factor * self.u_x_profile_3D_values #self.keil_profil(x[1], keil_percentage_of_inlet=self.keil_percentage_of_inlet, keil_u_max=self.units.characteristic_velocity_pu)

        #TODO: add temporal ramped equilibrium-boundary with profile (parameters: steps to ramp)
        else:
            raise NotImplementedError("Specify u_init = 0, 1, ...")

        return p, u

    @property
    def grid(self):
        return makeGrid(self.domain_constraints, self.shape)

    @property
    def boundaries(self):
        print("calling flow.boundaries()")
        time0 = time.time()

        # get grid and masks
        x, y, z = self.grid # in PU

        u_inlet_x = self.u_x_profile_3D_values[0, np.newaxis, ...] # self.keil_profil(y, keil_percentage_of_inlet=self.keil_percentage_of_inlet, keil_u_max=self.units.characteristic_velocity_pu)[0, np.newaxis,...]
        u_inlet_y = np.zeros_like(u_inlet_x)
        u_inlet = np.stack([u_inlet_x, u_inlet_y, u_inlet_y], axis=0)

        # INLET
        print("initializing inlet boundary condition...")
        if self.inlet_bc.casefold() == 'eqin':
            inlet_boundary_condition = EquilibriumBoundaryPU(self.in_mask, self.units.lattice, self.units, u_inlet)
        elif self.inlet_bc.casefold() == 'nex':
            inlet_boundary_condition = NonEquilibriumExtrapolationInletU(self.units.lattice, self.units, [-1, 0, 0],
                                                                         u_inlet)
        elif self.inlet_bc.casefold() == 'rampeqin':
            inlet_boundary_condition = RampedEquilibriumBoundaryPU(self.in_mask, self.units.lattice, self.units, u_inlet, ramp_steps=self.inlet_ramp_steps)
        else:
            print("(!) flow-class encountered illegal inlet_bc-parameter! Using EquilibriumBoundaryPU")
            inlet_boundary_condition = EquilibriumBoundaryPU(self.in_mask, self.units.lattice, self.units, u_inlet)

        # OUTLET
        print("initializing outlet boundary condition...")
        if self.outlet_bc.casefold() == 'eqoutp':
            outlet_boundary_condition = EquilibriumOutletP(self.units.lattice, [1, 0, 0], rho0=self.units.convert_pressure_pu_to_density_lu(0))
        elif self.outlet_bc.casefold() == 'eqoutu':
            print("(INFO) Equilibrium Outlet PU (!) was selected.")
            out_mask = np.zeros_like(self.in_mask)
            out_mask[-1, :, :] = True
            outlet_boundary_condition = EquilibriumBoundaryPU(out_mask, self.lattice, self.units, np.array(self.initial_solution(self.grid)[1])[0,...])
            #TODO: test the [0,...] indexing!
        else:  # default to EQ_outlet_P
            print("(INFO) outlet_bc was not recognized or specified. Defaulting to EQ_outlet_P")
            outlet_boundary_condition = EquilibriumOutletP(self.units.lattice, [1, 0, 0], rho0=self.units.convert_pressure_pu_to_density_lu(0))

        # LATERAL
        if self.lateral_bc == "periodic":
            pass
        else:
            raise NotImplementedError("lateral walls must be periodic or...")

        # LIST OF boundaries
        boundaries = [
            inlet_boundary_condition,
            outlet_boundary_condition
        ]
        # LATERAL NOT IMPLEMENTED YET

        i = 0
        for boundary in boundaries:
            print(f"boundaries[{i}]: {str(boundary)}")
            i += 1

        # time execution of flow.boundary()
        time1 = time.time() - time0
        print(f"boundaries took {floor(time1 / 60):02d}:{floor(time1 % 60):02d} [mm:ss].")
        return boundaries

    def u_x_profile_3D(self, y, inlet_velocity_pu):
        u_profile = np.zeros_like(y)
        if self.inlet_y_rel_start is not None and self.inlet_y_rel_end is not None:
            if int(y.shape[1]*self.inlet_y_rel_start) < int(y.shape[1]*self.inlet_y_rel_end):
                u_profile[:, int(y.shape[1]*self.inlet_y_rel_start):int(y.shape[1]*self.inlet_y_rel_end), :] = self.inlet_velocity_pu
            else:
                u_profile[:, int(y.shape[1]*self.inlet_y_rel_start), :] = self.inlet_velocity_pu
        elif self.inlet_y_rel_start is not None:
            u_profile[:, int(y.shape[1]*self.inlet_y_rel_start), :] = self.inlet_velocity_pu
        elif self.inlet_y_rel_end is not None:
            u_profile[:, int(y.shape[1] * self.inlet_y_rel_end), :] = self.inlet_velocity_pu
        else:
            u_profile[:, int(y.shape[1] * 0.5), :] = self.inlet_velocity_pu

        # TODO: test if keil-Profil works for INIT, EQ_in, EQ_out, because it might require 3xXxYxZ dims...
        return u_profile

