"""Lattice Boltzmann Solver"""

from timeit import default_timer as timer
from lettuce import (
    LettuceException, get_default_moment_transform, BGKInitialization, ExperimentalWarning, torch_gradient, HalfwayBounceBackBoundary, FullwayBounceBackBoundary,
)
from lettuce.util import pressure_poisson
import pickle
from copy import deepcopy
import warnings
import torch
import numpy as np

__all__ = ["Simulation"]


class Simulation:
    """High-level API for simulations.

    Attributes
    ----------
    reporters : list
        A list of reporters. Their call functions are invoked after every simulation step (and before the first one).

    """

    def __init__(self, flow, lattice, collision, streaming):
        self.flow = flow
        self.lattice = lattice
        self.collision = collision
        self.streaming = streaming
        self.i = 0  # Laufindex i für Schrittzahl
        # M.Bille: Kraftberechnung auf Objekt/BBB
        self.forceVal = []  # Liste der Kräfte (in x und y) über alle Schritte
        self.hwbb_present = False

        # CHECK initial solution for correct dimensions
        grid = flow.grid
        p, u = flow.initial_solution(grid)
        assert list(p.shape) == [1] + list(grid[0].shape), \
            LettuceException(f"Wrong dimension of initial pressure field. "
                             f"Expected {[1] + list(grid[0].shape)}, "
                             f"but got {list(p.shape)}.")
        assert list(u.shape) == [lattice.D] + list(grid[0].shape), \
            LettuceException("Wrong dimension of initial velocity field."
                             f"Expected {[lattice.D] + list(grid[0].shape)}, "
                             f"but got {list(u.shape)}.")

        # INITIALIZE distribution function f
        u = lattice.convert_to_tensor(flow.units.convert_velocity_to_lu(u))
        rho = lattice.convert_to_tensor(flow.units.convert_pressure_pu_to_density_lu(p))
        self.f = lattice.equilibrium(rho, lattice.convert_to_tensor(u))

        # list for reporters
        self.reporters = []

        # Define masks, where the collision or streaming are not applied
        # (initialized with 0, later specified by e.g. boundary conditions)
        x = flow.grid  # meshgrid, dimensions: D x nx x ny (x nz)
        self.no_collision_mask = lattice.convert_to_tensor(np.zeros_like(x[0], dtype=bool))  # dimensions: nx x ny (x nz)
        no_stream_mask = lattice.convert_to_tensor(np.zeros(self.f.shape, dtype=bool))
            # warum kein "self."? - vielleicht, weil es außerhalb der init nicht mehr gebraucht wird
            # ...(no_collision_mask wird für die collision gebraucht im call (s.u.), die no_streaming_mask wird aber direkt in der init
            # ...noch auf streaming.no_streaming_mask geschrieben (s.u.)

        # retrieve no-streaming and no-collision markings from all boundaries
        self._boundaries = deepcopy(self.flow.boundaries)  # store locally to keep the flow free from the boundary state
        for boundary in self._boundaries:
            if hasattr(boundary, "make_no_collision_mask"):
                # get no-collision markings from boundaries
                self.no_collision_mask = self.no_collision_mask | boundary.make_no_collision_mask(self.f.shape)
            if hasattr(boundary, "make_no_stream_mask"):
                # get no-streaming markings from boundaries
                no_stream_mask = no_stream_mask | boundary.make_no_stream_mask(self.f.shape)
        if no_stream_mask.any():
            # write no_streaming_mask to streaming-object
            self.streaming.no_stream_mask = no_stream_mask

        # define f_collided (post-collision, pre-streaming f), if HalfwayBounceBackBoundary is used
        for boundary in self._boundaries:
            if isinstance(boundary, HalfwayBounceBackBoundary):
                self.hwbb_present = True  # marks if Halfway Bounce Back Boundary is in use and f_collided is needed
                self.f_collided = deepcopy(self.f)

        # get pointer on obstacle_boundary for force_calculation
        self.obstacle_boundary = None
        if self._boundaries[-1] is not None:  # when obstacle == False, the obstacle_boundary is None
            self.obstacle_boundary = self._boundaries[-1]  # obst.b. should be the last boundary in the list

    def step(self, num_steps):
        """ Take num_steps stream-and-collision steps and return performance in MLUPS.
        M.Bille: added force_calculation on object/boundaries
        """
        start = timer()
        if self.i == 0:  # if this is the first timestep, calc. initial forceOnObject and call reporters
            # Perform force calculation on obstacle_boundary
            if self.obstacle_boundary is not None:
                self.forceVal.append(self.obstacle_boundary.calc_force_on_boundary(self.f))
            # reporters are called before the first timestep
            self._report()
        for _ in range(num_steps):  # simulate num_step timesteps
            ### COLLISION
            # Perform the collision routine everywhere, expect where the no_collision_mask is true
            # ...and store post-collision population for halfway-bounce-back boundary condition
            self.f = torch.where(self.no_collision_mask, self.f, self.collision(self.f))
            if self.hwbb_present:
                self.f_collided = deepcopy(self.f)

            ### CALCULATE FORCES ON OBSTACLE BOUNDARY
            if self.obstacle_boundary is not None:
                self.forceVal.append(self.obstacle_boundary.calc_force_on_boundary(self.f))

            ### STREAMING
            self.f = self.streaming(self.f)

            ### BOUNDARY
            # apply boundary conditions
            for boundary in self._boundaries:
                if boundary is not None:
                    if isinstance(boundary, HalfwayBounceBackBoundary):
                        self.f = boundary(self.f, self.f_collided)  # HalfwayBounceBackBoundary needs post-collision_pre-streaming f on boundary nodes to perform reflection of populations within the same timestep
                    else:
                        self.f = boundary(self.f)  # all non-HalfwayBounceBackBoundary-BoundaryConditions

            # count step
            self.i += 1

            # call reporters
            self._report()
        end = timer()

        # calculate runtime and performance in MLUPS
        seconds = end - start
        num_grid_points = self.lattice.rho(self.f).numel()
        mlups = num_steps * num_grid_points / 1e6 / seconds
        return mlups

    def _report(self):
        for reporter in self.reporters:
            reporter(self.i, self.flow.units.convert_time_to_pu(self.i), self.f)

    def initialize(self, max_num_steps=500, tol_pressure=0.001):
        """Iterative initialization to get moments consistent with the initial velocity.

        Using the initialization does not better TGV convergence. Maybe use a better scheme?
        """
        warnings.warn("Iterative initialization does not work well and solutions may diverge. Use with care. "
                      "Use initialize_f_neq instead.",
                      ExperimentalWarning)
        transform = get_default_moment_transform(self.lattice)
        collision = BGKInitialization(self.lattice, self.flow, transform)
        streaming = self.streaming
        p_old = 0
        for i in range(max_num_steps):
            self.f = streaming(self.f)
            self.f = collision(self.f)
            p = self.flow.units.convert_density_lu_to_pressure_pu(self.lattice.rho(self.f))
            if (torch.max(torch.abs(p - p_old))) < tol_pressure:
                break
            p_old = deepcopy(p)
        return i

    def initialize_pressure(self, max_num_steps=100000, tol_pressure=1e-6):
        """Reinitialize equilibrium distributions with pressure obtained by a Jacobi solver.
        Note that this method has to be called before initialize_f_neq.
        """
        u = self.lattice.u(self.f)
        rho = pressure_poisson(
            self.flow.units,
            self.lattice.u(self.f),
            self.lattice.rho(self.f),
            tol_abs=tol_pressure,
            max_num_steps=max_num_steps
        )
        self.f = self.lattice.equilibrium(rho, u)

    def initialize_f_neq(self):
        """Initialize the distribution function values. The f^(1) contributions are approximated by finite differences.
        See Krüger et al. (2017).
        """
        rho = self.lattice.rho(self.f)
        u = self.lattice.u(self.f)

        grad_u0 = torch_gradient(u[0], dx=1, order=6)[None, ...]
        grad_u1 = torch_gradient(u[1], dx=1, order=6)[None, ...]
        S = torch.cat([grad_u0, grad_u1])

        if self.lattice.D == 3:
            grad_u2 = torch_gradient(u[2], dx=1, order=6)[None, ...]
            S = torch.cat([S, grad_u2])

        Pi_1 = 1.0 * self.flow.units.relaxation_parameter_lu * rho * S / self.lattice.cs ** 2
        Q = (torch.einsum('ia,ib->iab', [self.lattice.e, self.lattice.e])
             - torch.eye(self.lattice.D, device=self.lattice.device, dtype=self.lattice.dtype) * self.lattice.cs ** 2)
        Pi_1_Q = self.lattice.einsum('ab,iab->i', [Pi_1, Q])
        fneq = self.lattice.einsum('i,i->i', [self.lattice.w, Pi_1_Q])

        feq = self.lattice.equilibrium(rho, u)
        self.f = feq - fneq

    def save_checkpoint(self, filename):
        """Write f as np.array using pickle module."""
        with open(filename, "wb") as fp:
            pickle.dump(self.f, fp)

    def load_checkpoint(self, filename):
        """Load f as np.array using pickle module."""
        with open(filename, "rb") as fp:
            self.f = pickle.load(fp)
