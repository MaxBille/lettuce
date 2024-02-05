"""Lattice Boltzmann Solver"""

from timeit import default_timer as timer
from lettuce import (
    LettuceException, get_default_moment_transform, BGKInitialization, ExperimentalWarning, torch_gradient,
    HalfwayBounceBackBoundary, FullwayBounceBackBoundary,
    InterpolatedBounceBackBoundary, InterpolatedBounceBackBoundary_compact_v1, InterpolatedBounceBackBoundary_compact_v2,
    FullwayBounceBackBoundary_compact, HalfwayBounceBackBoundary_compact_v1, HalfwayBounceBackBoundary_compact_v2,
    HalfwayBounceBackBoundary_compact_v3
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
        self.i = 0  # index of the current timestep

        # >>>
        # M.Bille:
        self.store_f_collided = []  # toggle if f is stored after collision and not overwritten through streaming,
        # ...f_collided might be needed together with f_collided_and_streamed for boundary-conditions or calculation of
        # ...momentum exchange (force on boundary, coefficient of drag etc.)
        # TRUE/FALSE per boundary in the _boundaries list

        self.times = [[], [], [], [], []]  # list of lists for time-measurement (collision, streaming, boundary, reporters)
        # ...doesn't work correctly with cuda-optimized code (e.g. c2 compact BBBC implementation)
        self.time_avg = dict()
        self.t_max = 72*3600-10*60  # max. runtime 71:50:00 h / to stop and store sim-data for later continuation,
        # ...because the cluster only allows 72h long jobs.
        # <<<

        # CALCULATE INITIAL SOLUTION of flow and CHECK initial solution for correct dimensions
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

        # INITIALIZE distribution function f: convert u and rho from numpy to torch.tensor
        u = lattice.convert_to_tensor(flow.units.convert_velocity_to_lu(u))
        rho = lattice.convert_to_tensor(flow.units.convert_pressure_pu_to_density_lu(p))
        self.f = lattice.equilibrium(rho, lattice.convert_to_tensor(u))

        # list of reporters
        self.reporters = []

        # Define masks, where collision or streaming are not applied
        # (initialized with 0, later specified by e.g. boundary conditions)
        x = flow.grid  # meshgrid, dimensions: D x nx x ny (x nz)
        self.no_collision_mask = lattice.convert_to_tensor(np.zeros_like(x[0], dtype=bool))  # dimensions: nx x ny (x nz)
        no_stream_mask = lattice.convert_to_tensor(np.zeros(self.f.shape, dtype=bool))
            # "self" and "no self" because no_stream_mask is written to streaming-object in the init,
            # ... no_collision_mask is used in the simulation.step()

        # retrieve no-streaming and no-collision markings from all boundaries
        self._boundaries = deepcopy(self.flow.boundaries)  # store locally to keep the flow free from the boundary state -> WHY?
        for boundary in self._boundaries:
            if hasattr(boundary, "make_no_collision_mask"):
                # get no-collision markings from boundaries
                self.no_collision_mask = self.no_collision_mask | boundary.make_no_collision_mask(self.f.shape)
            if hasattr(boundary, "make_no_stream_mask"):
                # get no-streaming markings from boundaries
                no_stream_mask = no_stream_mask | boundary.make_no_stream_mask(self.f.shape)
        if no_stream_mask.any():
            # pass no_streaming_mask to streaming-object
            self.streaming.no_stream_mask = no_stream_mask

        # define f_collided (post-collision, pre-streaming f) storage format for hwbb, ibb1, ibb1c1 or ibb1c2, ...
        self.boundary_range = range(len(self._boundaries))  # how many boundary condition python-objects
        for boundary_index in self.boundary_range:
            if hasattr(self._boundaries[boundary_index], "store_f_collided"):
                self.store_f_collided.append(True)  # this boundary needs f_collided
                self._boundaries[boundary_index].store_f_collided(self.f)
            else:
                self.store_f_collided.append(False)  # this boundary doesn't need f_collided
        # (!) at the moment f_collided is passed to and stored by the boundary. This is only efficient,
        # ...if there is either only one boundary needing f_collided, or the storage of f_collided is sparse,
        # ...meaning: every boundary stores only the population needed for itself.

    def step(self, num_steps):
        """ Take num_steps stream-and-collision steps and return performance in MLUPS.
        M.Bille: added force_calculation on object/boundaries
        M.Bille: added halfway bounce back boundary
        """
        start = timer()
        if self.i == 0:
            # reporters are called before the first timestep
            self._report()
        for _ in range(num_steps):  # simulate num_step timesteps
            time1 = timer()

            ### COLLISION
            # Perform the collision routine everywhere, expect where the no_collision_mask is true
            # ...and store post-collision population for certain bounce-back boundary condition
            self.f = torch.where(self.no_collision_mask, self.f, self.collision(self.f))

            time2 = timer()

            ### STORE f_collided FOR BOUNDARIES needing post-collision-, pre-streaming populations for bounce or force-calculation
            for boundary_index in self.boundary_range:
                if self.store_f_collided[boundary_index]:
                    self._boundaries[boundary_index].store_f_collided(self.f)

            time3 = timer()

            ### STREAMING
            self.f = self.streaming(self.f)

            time4 = timer()

            ### BOUNDARY
            # apply boundary conditions
            for boundary in self._boundaries:
                self.f = boundary(self.f)

            # count step
            self.i += 1

            time5 = timer()
            # call reporters
            self._report()

            time6 = timer()
            self.times[0].append(time2-time1)  # time to collide
            self.times[1].append(time3-time2)  # time to store f_collided
            self.times[2].append(time4-time3)  # time to stream
            self.times[3].append(time5-time4)  # time to boundary
            self.times[4].append(time6-time5)  # time to report

            # BETA: if simulation is running close to t_max (host job-duraction limit),
            # ...end simulation prematurely to allow for postprocessing and storage of so far gathered results.
            # If you suspect your simulation to end prematurely due to the execution time limit, remember to write a
            # ...checkpoint to continue the simulation in a new job.
            if time6-start > self.t_max:  # if T_total > 71:50:00 h
                num_steps = _  # log current step counter
                break  # end sim-loop prematurely
                # TODO: print out real number of steps! sim.i - i_start
        end = timer()

        # calculate individual runtimes (M.Bille): doesn't work with asynchronous, cuda-optimized boundary conditions
        if num_steps > 0:
            self.time_avg = dict(time_collision=sum(self.times[0])/len(self.times[0]),
                                 time_store_f_collided=sum(self.times[1])/len(self.times[1]),
                                 time_streaming=sum(self.times[2])/len(self.times[2]),
                                 time_boundary=sum(self.times[3])/len(self.times[3]),
                                 time_reporter=sum(self.times[4])/len(self.times[4]))
        else:  # no division by zero
            self.time_avg = dict(time_collision=-1,
                                 time_store_f_collided=-1,
                                 time_streaming=-1,
                                 time_boundary=-1,
                                 time_reporter=-1)

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

    def save_checkpoint(self, filename, device=None):
        """Write f as np.array using pickle module."""
        if device is not None:
            # to store checkpoint through cpu with "device='cpu'",
            # ...because on multi-gpu systems the checkpoint is loaded into the device-index it was stored on,
            # ...making multi-gpu multi-simulation runs tricky, because the device has to be ajusted after loading the
            # ...checkpoint. Recommend: copy f to cpu before checkpointing,
            # ...and loading back into gpu on loading checkpoint (see load_checkpoint() below)
            f_store = self.f.to(device, copy=True)
            with open(filename, "wb") as fp:
                pickle.dump(f_store, fp)
        else:  # if no device is given, checkpoint retains/contains device-affinity
            with open(filename, "wb") as fp:
                pickle.dump(self.f, fp)

    def load_checkpoint(self, filename, device=None):
        """Load f as np.array using pickle module."""
        if device is not None:
            with open(filename, "rb") as fp:
                f_load = pickle.load(fp)
                self.f = f_load.to(device, copy=False)
        else:  # if no device is given, device from checkponit is used. May run into issues if the device of the ckeckpoint is different from the device of the rest of the simulation.
            with open(filename, "rb") as fp:
                self.f = pickle.load(fp)
