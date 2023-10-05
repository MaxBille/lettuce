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

        # M.Bille:
        self.store_f_collided = "no"  # toggle if f is stored after collision and not overwritten through streaming,
        # ...f_collided might be needed together with f_collided_and_streamed for boundary-conditions or calculation of
        # ...momentum exchange (force on boundary, coefficient of drag etc.)
        self.times = [[], [], [], [], []]  # list of lists for time-measurement (collision, streaming, boundary, reporters)
        self.time_avg = dict()

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
            # write no_streaming_mask to streaming-object
            self.streaming.no_stream_mask = no_stream_mask

        # define f_collided (post-collision, pre-streaming f) storage format for hwbb, ibb1, ibb1c1 or ibb1c2, ...
        for boundary in self._boundaries:
            if isinstance(boundary, HalfwayBounceBackBoundary) or isinstance(boundary, InterpolatedBounceBackBoundary):
                self.store_f_collided = "dense"  # mark if a boundary is present which needs f_collided to be stored in dense format
            elif isinstance(boundary, InterpolatedBounceBackBoundary_compact_v1) or isinstance(boundary, HalfwayBounceBackBoundary_compact_v1):
                self.store_f_collided = "sparse"
            elif isinstance(boundary, HalfwayBounceBackBoundary_compact_v2):
                self.store_f_collided = "compact"
            elif isinstance(boundary, InterpolatedBounceBackBoundary_compact_v2):
                self.store_f_collided = "compact_ibb"

        if self.store_f_collided == "dense":
            self.f_collided = torch.clone(self.f)
        elif self.store_f_collided == "sparse":
            fc_q, fc_x, fc_y, fc_z = torch.where(self._boundaries[-1].f_mask + self._boundaries[-1].f_mask[self.lattice.stencil.opposite])
            self.fc_index = torch.stack((fc_q, fc_x, fc_y, fc_z))
            self.f_collided = torch.clone(torch.sparse_coo_tensor(indices=self.fc_index,
                                                               values=self.f[self.fc_index[0], self.fc_index[1],
                                                                             self.fc_index[2], self.fc_index[3]],
                                                               size=self.f.size()))
        elif self.store_f_collided == "compact":
            f_collided = torch.zeros_like(self._boundaries[-1].f_index[:,0], dtype=self.lattice.dtype)
            f_collided_opposite = torch.zeros_like(self._boundaries[-1].f_index[:,0], dtype=self.lattice.dtype)
            if self.lattice.D == 2:
                f_collided_lt = deepcopy(self.f[self._boundaries[-1].f_index[:, 0],  # q
                                                self._boundaries[-1].f_index[:, 1],  # x
                                                self._boundaries[-1].f_index[:, 2]])  # y
                f_collided_lt_opposite = deepcopy(self.f[self._boundaries[-1].opposite_tensor[
                                                             self._boundaries[-1].f_index[:,0]],  # q
                                                         self._boundaries[-1].f_index[:, 1],  # x
                                                         self._boundaries[-1].f_index[:, 2]])  # y
            if self.lattice.D == 3:
                f_collided_lt = deepcopy(self.f[self._boundaries[-1].f_index[:, 0],  # q
                                                self._boundaries[-1].f_index[:, 1],  # x
                                                self._boundaries[-1].f_index[:, 2],  # y
                                                self._boundaries[-1].f_index[:, 3]])  # z
                f_collided_lt_opposite = deepcopy(self.f[self._boundaries[-1].opposite_tensor[
                                                             self._boundaries[-1].f_index[:,0]],  # q
                                                         self._boundaries[-1].f_index[:, 1],  # x
                                                         self._boundaries[-1].f_index[:, 2],  # y
                                                         self._boundaries[-1].f_index[:, 3]])  # z
            self.f_collided = torch.stack((f_collided, f_collided_opposite), dim=1)
        elif self.store_f_collided == "compact_ibb":
            f_collided_lt = torch.zeros_like(self._boundaries[-1].d_lt)  # float-tensor with number of (x_b nodes with d<=0.5) values
            f_collided_gt = torch.zeros_like(self._boundaries[-1].d_gt)  # float-tensor with number of (x_b nodes with d>0.5) values
            f_collided_lt_opposite = torch.zeros_like(self._boundaries[-1].d_lt)
            f_collided_gt_opposite = torch.zeros_like(self._boundaries[-1].d_gt)

            if self.lattice.D == 2:
                f_collided_lt = deepcopy(self.f[self._boundaries[-1].f_index_lt[:, 0],  # q
                                                self._boundaries[-1].f_index_lt[:, 1],  # x
                                                self._boundaries[-1].f_index_lt[:, 2]])  # y
                f_collided_lt_opposite = deepcopy(self.f[self._boundaries[-1].opposite_tensor[
                                                             self._boundaries[-1].f_index_lt[:,0]],  # q
                                                         self._boundaries[-1].f_index_lt[:, 1],  # x
                                                         self._boundaries[-1].f_index_lt[:, 2]])  # y

                f_collided_gt = deepcopy(self.f[self._boundaries[-1].f_index_gt[:, 0],  # q
                                                self._boundaries[-1].f_index_gt[:, 1],  # x
                                                self._boundaries[-1].f_index_gt[:, 2]])  # y
                f_collided_gt_opposite = deepcopy(self.f[self._boundaries[-1].opposite_tensor[
                                                             self._boundaries[-1].f_index_gt[:,0]],  # q
                                                         self._boundaries[-1].f_index_gt[:, 1],  # x
                                                         self._boundaries[-1].f_index_gt[:, 2]])  # y
            if self.lattice.D == 3:
                f_collided_lt = deepcopy(self.f[self._boundaries[-1].f_index_lt[:, 0],  # q
                                                self._boundaries[-1].f_index_lt[:, 1],  # x
                                                self._boundaries[-1].f_index_lt[:, 2],  # y
                                                self._boundaries[-1].f_index_lt[:, 3]])  # z
                f_collided_lt_opposite = deepcopy(self.f[self._boundaries[-1].opposite_tensor[
                                                             self._boundaries[-1].f_index_lt[:,0]],  # q
                                                         self._boundaries[-1].f_index_lt[:, 1],  # x
                                                         self._boundaries[-1].f_index_lt[:, 2],  # y
                                                         self._boundaries[-1].f_index_lt[:, 3]])  # z

                f_collided_gt = deepcopy(self.f[self._boundaries[-1].f_index_gt[:, 0],  # q
                                                self._boundaries[-1].f_index_gt[:, 1],  # x
                                                self._boundaries[-1].f_index_gt[:, 2],  # y
                                                self._boundaries[-1].f_index_gt[:, 3]])  # z
                f_collided_gt_opposite = deepcopy(self.f[self._boundaries[-1].opposite_tensor[
                                                             self._boundaries[-1].f_index_gt[:,0]],  # q
                                                         self._boundaries[-1].f_index_gt[:, 1],  # x
                                                         self._boundaries[-1].f_index_gt[:, 2],  # y
                                                         self._boundaries[-1].f_index_gt[:, 3]])  # z
            # f_collided in compact storage-format (because torch.to_sparse() doesn't allow value-assignment and/or batch-indexing)
            self.f_collided_lt = torch.stack((f_collided_lt, f_collided_lt_opposite), dim=1)
            self.f_collided_gt = torch.stack((f_collided_gt, f_collided_gt_opposite), dim=1)

    def step(self, num_steps):
        """ Take num_steps stream-and-collision steps and return performance in MLUPS.
        M.Bille: added force_calculation on object/boundaries
        M.Bille: added halfway bounce back boundary
        """
        start = timer()
        if self.i == 0:  # if this is the first timestep, calc. initial force on Object/walls/boundary/obstacle and call reporters
            # reporters are called before the first timestep
            self._report()
        for _ in range(num_steps):  # simulate num_step timesteps
            time1 = timer()
            ### COLLISION
            # Perform the collision routine everywhere, expect where the no_collision_mask is true
            # ...and store post-collision population for halfway-bounce-back boundary condition
            self.f = torch.where(self.no_collision_mask, self.f, self.collision(self.f))

            time2 = timer()
            if self.store_f_collided == "dense":  # f_collided in regular full dense storage format
                self.f_collided = torch.clone(self.f)
            elif self.store_f_collided == "sparse":  # f_collided in torch.sparse() storage format
                self.f_collided = torch.clone(torch.sparse_coo_tensor(indices=self.fc_index,
                                                                   values=self.f[self.fc_index[0], self.fc_index[1],
                                                                                 self.fc_index[2], self.fc_index[3]],
                                                                   size=self.f.size()))
            elif self.store_f_collided == "compact":  # f_collided in super compact storage format
                if self.lattice.D == 2:
                    self.f_collided[:, 0] = torch.clone(self.f[self._boundaries[-1].f_index[:, 0],  # q
                                                                  self._boundaries[-1].f_index[:, 1],  # x
                                                                  self._boundaries[-1].f_index[:, 2]])  # y
                    self.f_collided[:, 1] = torch.clone(self.f[self._boundaries[-1].opposite_tensor[
                                                                      self._boundaries[-1].f_index[:,0]],  # q
                                                                  self._boundaries[-1].f_index[:, 1],  # x
                                                                  self._boundaries[-1].f_index[:, 2]])  # y
                if self.lattice.D == 3:
                    self.f_collided[:, 0] = torch.clone(self.f[self._boundaries[-1].f_index[:, 0],  # q
                                                                  self._boundaries[-1].f_index[:, 1],  # x
                                                                  self._boundaries[-1].f_index[:, 2],  # y
                                                                  self._boundaries[-1].f_index[:, 3]])  # z
                    self.f_collided[:, 1] = torch.clone(self.f[self._boundaries[-1].opposite_tensor[
                                                                      self._boundaries[-1].f_index[:,0]],  # q
                                                                  self._boundaries[-1].f_index[:, 1],  # x
                                                                  self._boundaries[-1].f_index[:, 2],  # y
                                                                  self._boundaries[-1].f_index[:, 3]])  # z
            elif self.store_f_collided == "compact_ibb":  # f_collided in super compact storage format for ibb
                if self.lattice.D == 2:
                    self.f_collided_lt[:, 0] = torch.clone(self.f[self._boundaries[-1].f_index_lt[:, 0],  # q
                                                                  self._boundaries[-1].f_index_lt[:, 1],  # x
                                                                  self._boundaries[-1].f_index_lt[:, 2]])  # y
                    self.f_collided_lt[:, 1] = torch.clone(self.f[self._boundaries[-1].opposite_tensor[
                                                                      self._boundaries[-1].f_index_lt[:,0]],  # q
                                                                  self._boundaries[-1].f_index_lt[:, 1],  # x
                                                                  self._boundaries[-1].f_index_lt[:, 2]])  # y

                    self.f_collided_gt[:, 0] = torch.clone(self.f[self._boundaries[-1].f_index_gt[:, 0],  # q
                                                                  self._boundaries[-1].f_index_gt[:, 1],  # x
                                                                  self._boundaries[-1].f_index_gt[:, 2]])  # y
                    self.f_collided_gt[:, 1] = torch.clone(self.f[self._boundaries[-1].opposite_tensor[
                                                                      self._boundaries[-1].f_index_gt[:,0]],  # q
                                                                  self._boundaries[-1].f_index_gt[:, 1],  # x
                                                                  self._boundaries[-1].f_index_gt[:, 2]])  # y
                if self.lattice.D == 3:
                    self.f_collided_lt[:, 0] = torch.clone(self.f[self._boundaries[-1].f_index_lt[:, 0],  # q
                                                                  self._boundaries[-1].f_index_lt[:, 1],  # x
                                                                  self._boundaries[-1].f_index_lt[:, 2],  # y
                                                                  self._boundaries[-1].f_index_lt[:, 3]])  # z
                    self.f_collided_lt[:, 1] = torch.clone(self.f[self._boundaries[-1].opposite_tensor[
                                                                      self._boundaries[-1].f_index_lt[:,0]],  # q
                                                                  self._boundaries[-1].f_index_lt[:, 1],  # x
                                                                  self._boundaries[-1].f_index_lt[:, 2],  # y
                                                                  self._boundaries[-1].f_index_lt[:, 3]])  # z

                    self.f_collided_gt[:, 0] = torch.clone(self.f[self._boundaries[-1].f_index_gt[:, 0],  # q
                                                                  self._boundaries[-1].f_index_gt[:, 1],  # x
                                                                  self._boundaries[-1].f_index_gt[:, 2],  # y
                                                                  self._boundaries[-1].f_index_gt[:, 3]])  # z
                    self.f_collided_gt[:, 1] = torch.clone(self.f[self._boundaries[-1].opposite_tensor[
                                                                      self._boundaries[-1].f_index_gt[:,0]],  # q
                                                                  self._boundaries[-1].f_index_gt[:, 1],  # x
                                                                  self._boundaries[-1].f_index_gt[:, 2],  # y
                                                                  self._boundaries[-1].f_index_gt[:, 3]])  # z
            time3 = timer()
            ### STREAMING
            self.f = self.streaming(self.f)

            time4 = timer()
            ### BOUNDARY
            # apply boundary conditions
            for boundary in self._boundaries:
                if boundary is not None:
                    if isinstance(boundary, HalfwayBounceBackBoundary) \
                            or isinstance(boundary, InterpolatedBounceBackBoundary) \
                            or isinstance(boundary, InterpolatedBounceBackBoundary_compact_v1) \
                            or isinstance(boundary, HalfwayBounceBackBoundary_compact_v1) \
                            or isinstance(boundary, HalfwayBounceBackBoundary_compact_v2):
                        self.f = boundary(self.f, self.f_collided)  # boundary needs post-collision_pre-streaming f on boundary nodes to perform reflection of populations within the same timestep
                    elif isinstance(boundary, InterpolatedBounceBackBoundary_compact_v2):
                        self.f = boundary(self.f, self.f_collided_lt, self.f_collided_gt)  # f_collided in compact storage format
                    else:
                        self.f = boundary(self.f)  # all BC which do not use any other populations

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
        end = timer()

        # calculate individual runtimes (M.Bille)
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

    def save_checkpoint(self, filename):
        """Write f as np.array using pickle module."""
        with open(filename, "wb") as fp:
            pickle.dump(self.f, fp)

    def load_checkpoint(self, filename):
        """Load f as np.array using pickle module."""
        with open(filename, "rb") as fp:
            self.f = pickle.load(fp)
