class InterpolatedBounceBackBoundary:

    def __init__(self, mask, lattice, x_center, y_center, radius, interpolation_order=1):
        log_VRAM("IBB1_init (start)")
        t_init_start = time.time()
        self.interpolation_order = interpolation_order
        self.mask = mask  # location of solid-nodes
        self.lattice = lattice
        self.force_sum = torch.zeros_like(self.lattice.convert_to_tensor(
            self.lattice.stencil.e[0]))  # summed force vector on all boundary nodes, in D dimensions (x,y,(z))
        ### create f_mask, needed for force-calculation
        # ...(marks all fs which point from fluid to solid (boundary) and considered for momentum exchange)

        self.f_mask_sequential_gt = []  # coordinates (q,x,y,[z]) for all force- and bounce-relevant populations with d<=0.5
        self.f_mask_sequential_lt = []  # coordinates (q,x,y,[z]) for all force- and bounce-relevant populations with d>0.5
        self.d_sequential_lt = []  # relative "d"istance between the solid node and the ideal boundary location for d<=0.5
        self.d_sequential_gt = []  # relative "d"istance between the solid node and the ideal boundary location for d>0.5

        if self.lattice.D == 2:
            nx, ny = mask.shape  # domain size in x and y
            a, b = np.where(mask)
            # np.arrays: list of (a) x-coordinates and (b) y-coordinates in the boundary.mask
            # ...to enable iteration over all boundary/wall/object-nodes
            for p in range(0, len(a)):  # for all TRUE-nodes in boundary.mask
                for i in range(0, self.lattice.Q):  # for all stencil-directions c_i (lattice.stencil.e in lettuce)
                    # check for boundary-nodes neighboring the domain-border.
                    # ...they have to take the periodicity into account...
                    border = np.zeros(self.lattice.D, dtype=int)
                    if a[p] == 0 and self.lattice.stencil.e[i, 0] == -1:  # searching border on left
                        border[0] = -1
                    elif a[p] == nx - 1 and self.lattice.e[i, 0] == 1:  # searching border on right
                        border[0] = 1
                    if b[p] == 0 and self.lattice.stencil.e[i, 1] == -1:  # searching border on left
                        border[1] = -1
                    elif b[p] == ny - 1 and self.lattice.e[i, 1] == 1:  # searching border on right
                        border[1] = 1
                    try:  # try in case the neighboring cell does not exist (= an f pointing out of the simulation domain)
                        if not mask[a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                    b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny]:
                            # if the neighbour of p is False in the boundary.mask, p is a solid node, neighbouring a fluid node:
                            # ...the direction pointing from the fluid neighbour to solid p is marked on the (fluid) neighbour,
                            # ...(needed for bounce-back (hwbb and ibb1) and force-calculation)

                            # calculate intersection point of boundary surface and link ->
                            # ...calculate distance between fluid node and boundary surface on the link
                            px = a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx  # fluid node x-coordinate
                            py = b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny  # fluid node y-coordinate
                            cx = self.lattice.stencil.e[
                                self.lattice.stencil.opposite[i], 0]  # link-direction x to solid node
                            cy = self.lattice.stencil.e[
                                self.lattice.stencil.opposite[i], 1]  # link-direction y to solid node

                            # pq-formula
                            h1 = (px * cx + py * cy - cx * x_center - cy * y_center) / (cx * cx + cy * cy)  # p/2
                            h2 = (px * px + py * py + x_center * x_center + y_center * y_center
                                  - 2 * px * x_center - 2 * py * y_center - radius * radius) / (
                                         cx * cx + cy * cy)  # q

                            d1 = - h1 + np.sqrt(h1 * h1 - h2)
                            d2 = - h1 - np.sqrt(h1 * h1 - h2)

                            # print("xb,yb,i,d1,d2 xf, yf, cx, cy:", a[p], b[p],i,d1,d2,px,py,cx,cy)

                            # distance (LU) from fluid node to the "true" boundary location
                            if d1 <= 1 and np.isreal(d1):  # d should be between 0 and 1
                                if d1 <= 0.5:
                                    self.d_sequential_lt.append(d1)
                                    self.f_mask_sequential_lt.append([self.lattice.stencil.opposite[i],
                                                                      a[p] + self.lattice.stencil.e[i, 0] - border[
                                                                          0] * nx,
                                                                      b[p] + self.lattice.stencil.e[i, 1] - border[
                                                                          1] * ny])
                                else:
                                    self.d_sequential_gt.append(d1)
                                    self.f_mask_sequential_gt.append([self.lattice.stencil.opposite[i],
                                                                      a[p] + self.lattice.stencil.e[i, 0] - border[
                                                                          0] * nx,
                                                                      b[p] + self.lattice.stencil.e[i, 1] - border[
                                                                          1] * ny])
                            elif d2 <= 1 and np.isreal(d2):
                                if d2 <= 0.5:
                                    self.d_sequential_lt.append(d2)
                                    self.f_mask_sequential_lt.append([self.lattice.stencil.opposite[i],
                                                                      a[p] + self.lattice.stencil.e[i, 0] - border[
                                                                          0] * nx,
                                                                      b[p] + self.lattice.stencil.e[i, 1] - border[
                                                                          1] * ny])
                                else:
                                    self.d_sequential_gt.append(d2)
                                    self.f_mask_sequential_gt.append([self.lattice.stencil.opposite[i],
                                                                      a[p] + self.lattice.stencil.e[i, 0] - border[
                                                                          0] * nx,
                                                                      b[p] + self.lattice.stencil.e[i, 1] - border[
                                                                          1] * ny])
                            else:  # if both distances d1, d2 are not between 0 and 1
                                print("IBB WARNING: d1 is", d1, "; d2 is", d2, "for boundaryPoint x,y,ci", a[p],
                                      b[p], self.lattice.stencil.e[i, 0], self.lattice.stencil.e[i, 1])
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there
        if self.lattice.D == 3:  # like 2D, but in 3D...guess what...
            nx, ny, nz = mask.shape
            a, b, c = np.where(mask)
            for p in range(0, len(a)):
                for i in range(0, self.lattice.Q):
                    border = np.zeros(self.lattice.D, dtype=int)
                    if a[p] == 0 and self.lattice.stencil.e[i, 0] == -1:  # searching border on left
                        border[0] = -1
                    elif a[p] == nx - 1 and self.lattice.e[i, 0] == 1:  # searching border on right
                        border[0] = 1
                    if b[p] == 0 and self.lattice.stencil.e[i, 1] == -1:  # searching border on left
                        border[1] = -1
                    elif b[p] == ny - 1 and self.lattice.e[i, 1] == 1:  # searching border on right
                        border[1] = 1
                    if c[p] == 0 and self.lattice.stencil.e[i, 2] == -1:  # searching border on left
                        border[2] = -1
                    elif c[p] == nz - 1 and self.lattice.e[i, 2] == 1:  # searching border on right
                        border[2] = 1
                    try:  # try in case the neighboring cell does not exist (an f pointing out of simulation domain)
                        if not mask[a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                    b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                    c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz]:

                            # calculate intersection point of boundary surface and link ->
                            # ...calculate distance between fluid node and boundary surface on the link
                            px = a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx  # fluid node x-coordinate
                            py = b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny  # fluid node y-coordinate
                            # Z-coodinate not needed for cylinder ! #pz = c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz  # fluid node z-coordinate

                            cx = self.lattice.stencil.e[
                                self.lattice.stencil.opposite[i], 0]  # link-direction x to solid node
                            cy = self.lattice.stencil.e[
                                self.lattice.stencil.opposite[i], 1]  # link-direction y to solid node
                            # Z-coodinate not needed for cylinder ! #cz = self.lattice.stencil.e[
                            #    self.lattice.stencil.opposite[i], 2]  # link-direction z to solid node

                            # pq-formula
                            h1 = (px * cx + py * cy - cx * x_center - cy * y_center) / (cx * cx + cy * cy)  # p/2
                            h2 = (px * px + py * py + x_center * x_center + y_center * y_center
                                  - 2 * px * x_center - 2 * py * y_center - radius * radius) / (
                                         cx * cx + cy * cy)  # q

                            d1 = - h1 + np.sqrt(h1 * h1 - h2)
                            d2 = - h1 - np.sqrt(h1 * h1 - h2)

                            # print("xb,yb,i,d1,d2 xf, yf, cx, cy:", a[p], b[p], i, d1, d2, px, py, cx, cy)
                            # distance (LU) from fluid node to the "true" boundary location
                            if d1 <= 1 and np.isreal(d1):  # d should be between 0 and 1
                                if d1 <= 0.5:
                                    self.d_sequential_lt.append(d1)
                                    self.f_mask_sequential_lt.append([self.lattice.stencil.opposite[i],
                                                                      a[p] + self.lattice.stencil.e[i, 0] - border[
                                                                          0] * nx,
                                                                      b[p] + self.lattice.stencil.e[i, 1] - border[
                                                                          1] * ny,
                                                                      c[p] + self.lattice.stencil.e[i, 2] - border[
                                                                          2] * nz])
                                else:
                                    self.d_sequential_gt.append(d1)
                                    self.f_mask_sequential_gt.append([self.lattice.stencil.opposite[i],
                                                                      a[p] + self.lattice.stencil.e[i, 0] - border[
                                                                          0] * nx,
                                                                      b[p] + self.lattice.stencil.e[i, 1] - border[
                                                                          1] * ny,
                                                                      c[p] + self.lattice.stencil.e[i, 2] - border[
                                                                          2] * nz])
                            elif d2 <= 1 and np.isreal(d2):
                                if d2 <= 0.5:
                                    self.d_sequential_lt.append(d2)
                                    self.f_mask_sequential_lt.append([self.lattice.stencil.opposite[i],
                                                                      a[p] + self.lattice.stencil.e[i, 0] - border[
                                                                          0] * nx,
                                                                      b[p] + self.lattice.stencil.e[i, 1] - border[
                                                                          1] * ny,
                                                                      c[p] + self.lattice.stencil.e[i, 2] - border[
                                                                          2] * nz])
                                else:
                                    self.d_sequential_gt.append(d2)
                                    self.f_mask_sequential_gt.append([self.lattice.stencil.opposite[i],
                                                                      a[p] + self.lattice.stencil.e[i, 0] - border[
                                                                          0] * nx,
                                                                      b[p] + self.lattice.stencil.e[i, 1] - border[
                                                                          1] * ny,
                                                                      c[p] + self.lattice.stencil.e[i, 2] - border[
                                                                          2] * nz])
                            else:  # if both distances d1, d2 are not between 0 and 1
                                print("IBB WARNING: d1 is", d1, "; d2 is", d2, "for boundaryPoint x,y,ci", a[p],
                                      b[p], c[p], self.lattice.stencil.e[i, 0], self.lattice.stencil.e[i, 1],
                                      self.lattice.stencil.e[i, 2])
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there

        self.f_mask_sequential_lt = torch.tensor(np.array(self.f_mask_sequential_lt), device=self.lattice.device,
                                                 dtype=torch.int64)  # the index has has to be integer
        self.f_mask_sequential_gt = torch.tensor(np.array(self.f_mask_sequential_gt), device=self.lattice.device,
                                                 dtype=torch.int64)  # the index has has to be integer

        self.d_sequential_lt = self.lattice.convert_to_tensor(np.array(self.d_sequential_lt))
        self.d_sequential_gt = self.lattice.convert_to_tensor(np.array(self.d_sequential_gt))

        self.opposite_tensor = torch.tensor(self.lattice.stencil.opposite, device=self.lattice.device,
                                            dtype=torch.int64)

        print("IBB initialization took " + str(time.time() - t_init_start) + "seconds")
        log_VRAM("IBB1_init (end)")

    def __call__(self, f, f_collided_lt, f_collided_gt):
        log_VRAM("IBB1_call (start)")
        log_VRAM("IBB1_call (PLACEHOLDER f_tmp)")
        ## f_collided_lt = [f_collided_lt, f_collided_lt.opposite] (!) in compact storage-layout

        if self.lattice.D == 2:
            # BOUNCE
            # if d <= 0.5
            f[self.opposite_tensor[self.f_mask_sequential_lt[:, 0]],
              self.f_mask_sequential_lt[:, 1],
              self.f_mask_sequential_lt[:, 2]] = 2 * self.d_sequential_lt * f_collided_lt[:, 0] \
                                                 + (1 - 2 * self.d_sequential_lt) * f[
                                                     self.f_mask_sequential_lt[:, 0],
                                                     self.f_mask_sequential_lt[:, 1],
                                                     self.f_mask_sequential_lt[:, 2]]
            log_VRAM("IBB1_call (after_bounce_lt)")
            # if d > 0.5
            f[self.opposite_tensor[self.f_mask_sequential_gt[:, 0]],
              self.f_mask_sequential_gt[:, 1],
              self.f_mask_sequential_gt[:, 2]] = (1 / (2 * self.d_sequential_gt)) * f_collided_gt[:, 0] \
                                                 + (1 - 1 / (2 * self.d_sequential_gt)) * f_collided_gt[:, 1]

            log_VRAM("IBB1_call (after_bounce_gt)")
        if self.lattice.D == 3:
            # BOUNCE
            # if d <= 0.5
            f[self.opposite_tensor[self.f_mask_sequential_lt[:, 0]],
              self.f_mask_sequential_lt[:, 1],
              self.f_mask_sequential_lt[:, 2],
              self.f_mask_sequential_lt[:, 3]] = 2 * self.d_sequential_lt * f_collided_lt[:, 0] \
                                                 + (1 - 2 * self.d_sequential_lt) * f[
                                                     self.f_mask_sequential_lt[:, 0],
                                                     self.f_mask_sequential_lt[:, 1],
                                                     self.f_mask_sequential_lt[:, 2],
                                                     self.f_mask_sequential_lt[:, 3]]
            log_VRAM("IBB1_call (after_bounce_lt)")
            # if d > 0.5
            f[self.opposite_tensor[self.f_mask_sequential_gt[:, 0]],
              self.f_mask_sequential_gt[:, 1],
              self.f_mask_sequential_gt[:, 2],
              self.f_mask_sequential_gt[:, 3]] = (1 / (2 * self.d_sequential_gt)) * f_collided_gt[:, 0] \
                                                 + (1 - 1 / (2 * self.d_sequential_gt)) * f_collided_gt[:, 1]

            log_VRAM("IBB1_call (after_bounce_gt)")
        log_VRAM("IBB1_call (PLACEHOLDER after_bounce)")
        # CALC. FORCE on boundary (MEM, MEA)
        self.calc_force_on_boundary(f, f_collided_lt, f_collided_gt)

        log_VRAM("IBB1_call (end, after calcForce, before 'return')")
        return f

    def make_no_stream_mask(self, f_shape):
        assert self.mask.shape == f_shape[1:]  # all dimensions of f except the 0th (q)
        # no_stream_mask has to be dimensions: (q,x,y,z) (z optional), but CAN be (x,y,z) (z optional).
        # ...in the latter case, torch.where broadcasts the mask to (q,x,y,z), so ALL q populations of a lattice-node are marked equally
        return self.lattice.convert_to_tensor(self.mask)

    def make_no_collision_mask(self, f_shape):
        # INFO: for the halfway bounce back boundary, a no_collision_mask ist not necessary, because the no_streaming_mask
        # ...prevents interaction between nodes inside and outside of the boundary region.
        # INFO: pay attention to the initialization of observable/moment-fields (u, rho,...) on the boundary nodes,
        # ...in the initial solution of your flow, especially if visualization or post processing uses the field-values
        # ...in the whole domain (including the boundary region)!
        assert self.mask.shape == f_shape[1:]
        return self.lattice.convert_to_tensor(self.mask)

    def calc_force_on_boundary(self, f_bounced, f_collided_lt, f_collided_gt):
        log_VRAM("IBB1_force (start)")
        log_VRAM("IBB1_force (PLACEHOLDER tmp)")
        ### force = e * (f_c + f_b[opp.])
        if self.lattice.D == 2:
            self.force_sum = torch.einsum('i..., id -> d',
                                          f_collided_lt[:, 0] + f_bounced[
                                              self.opposite_tensor[self.f_mask_sequential_lt[:, 0]],
                                              self.f_mask_sequential_lt[:, 1],
                                              self.f_mask_sequential_lt[:, 2]],
                                          self.lattice.e[self.f_mask_sequential_lt[:, 0]]) \
                             + torch.einsum('i..., id -> d',
                                            f_collided_gt[:, 0] + f_bounced[
                                                self.opposite_tensor[self.f_mask_sequential_gt[:, 0]],
                                                self.f_mask_sequential_gt[:, 1],
                                                self.f_mask_sequential_gt[:, 2]],
                                            self.lattice.e[self.f_mask_sequential_gt[:, 0]])
        if self.lattice.D == 3:
            self.force_sum = torch.einsum('i..., id -> d',
                                          f_collided_lt[:, 0] + f_bounced[
                                              self.opposite_tensor[self.f_mask_sequential_lt[:, 0]],
                                              self.f_mask_sequential_lt[:, 1],
                                              self.f_mask_sequential_lt[:, 2],
                                              self.f_mask_sequential_lt[:, 3]],
                                          self.lattice.e[self.f_mask_sequential_lt[:, 0]]) \
                             + torch.einsum('i..., id -> d',
                                            f_collided_gt[:, 0] + f_bounced[
                                                self.opposite_tensor[self.f_mask_sequential_gt[:, 0]],
                                                self.f_mask_sequential_gt[:, 1],
                                                self.f_mask_sequential_gt[:, 2],
                                                self.f_mask_sequential_gt[:, 3]],
                                            self.lattice.e[self.f_mask_sequential_gt[:, 0]])

        log_VRAM("IBB1_force (end)")


class Simulation:

    def __init__(self, flow, lattice, collision, streaming):
        log_VRAM("SIM_init (start)")
        self.flow = flow
        self.lattice = lattice
        self.collision = collision
        self.streaming = streaming
        self.i = 0  # index of the current timestep

        # M.Bille:
        self.store_f_collided = False  # toggle if f is stored after collision and not overwritten through streaming,
        # ...f_collided might be needed together with f_collided_and_streamed for boundary-conditions or calculation of
        # ...momentum exchange (force on boundary, coefficient of drag etc.)
        self.store_f_collided_compact = False  # toggle compact storage of f_collided for interpolated_bounce_back boundary
        self.times = [[], [], [],
                      []]  # list of lists for time-measurement (collision, streaming, boundary, reporters)
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
        log_VRAM("SIM_init (flow.initial_solution)")
        u = lattice.convert_to_tensor(flow.units.convert_velocity_to_lu(u))
        rho = lattice.convert_to_tensor(flow.units.convert_pressure_pu_to_density_lu(p))
        log_VRAM("SIM_init (u,rho to tensor)")
        self.f = lattice.equilibrium(rho, lattice.convert_to_tensor(u))
        log_VRAM("SIM_init (f created)")

        # list of reporters
        self.reporters = []

        # Define masks, where collision or streaming are not applied
        # (initialized with 0, later specified by e.g. boundary conditions)
        x = flow.grid  # meshgrid, dimensions: D x nx x ny (x nz)
        self.no_collision_mask = lattice.convert_to_tensor(
            np.zeros_like(x[0], dtype=bool))  # dimensions: nx x ny (x nz)
        log_VRAM("SIM_init (no_collision_mask)")
        no_stream_mask = lattice.convert_to_tensor(np.zeros(self.f.shape, dtype=bool))
        log_VRAM("SIM_init (no_streaming_mask)")
        # "self" and "no self" because no_stream_mask is written to streaming-object in the init,
        # ... no_collision_mask is used in the simulation.step()

        # retrieve no-streaming and no-collision markings from all boundaries
        self._boundaries = deepcopy(
            self.flow.boundaries)  # store locally to keep the flow free from the boundary state -> WHY?
        log_VRAM("SIM_init (flow.boundaries)")
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
        log_VRAM("SIM_init (NS_mask to streaming)")

        # define f_collided (post-collision, pre-streaming f), if HalfwayBounceBackBoundary is used
        for boundary in self._boundaries:
            if isinstance(boundary, HalfwayBounceBackBoundary) or isinstance(boundary,
                                                                             InterpolatedBounceBackBoundary):
                self.store_f_collided = True  # mark if a boundary is present which needs f_collided to be stored
                # self.f_collided = deepcopy(self.f)
            if isinstance(boundary, InterpolatedBounceBackBoundary):
                self.store_f_collided_compact = True
        if self.store_f_collided:
            if self.store_f_collided_compact:
                f_collided_lt = torch.zeros_like(self._boundaries[
                                                     -1].d_sequential_lt)  # float-tensor with number of (x_b nodes with d<=0.5) values
                f_collided_gt = torch.zeros_like(self._boundaries[
                                                     -1].d_sequential_gt)  # float-tensor with number of (x_b nodes with d>0.5) values
                f_collided_lt_opposite = torch.zeros_like(self._boundaries[-1].d_sequential_lt)
                f_collided_gt_opposite = torch.zeros_like(self._boundaries[-1].d_sequential_gt)
                log_VRAM("SIM_init (f_collided subtensors created)")

                if self.lattice.D == 2:
                    f_collided_lt = deepcopy(self.f[self._boundaries[-1].f_mask_sequential_lt[:, 0],  # q
                                                    self._boundaries[-1].f_mask_sequential_lt[:, 1],  # x
                                                    self._boundaries[-1].f_mask_sequential_lt[:, 2]])  # y
                    f_collided_lt_opposite = deepcopy(self.f[self._boundaries[-1].opposite_tensor[
                                                                 self._boundaries[-1].f_mask_sequential_lt[:,
                                                                 0]],  # q
                                                             self._boundaries[-1].f_mask_sequential_lt[:, 1],  # x
                                                             self._boundaries[-1].f_mask_sequential_lt[:, 2]])  # y

                    f_collided_gt = deepcopy(self.f[self._boundaries[-1].f_mask_sequential_gt[:, 0],  # q
                                                    self._boundaries[-1].f_mask_sequential_gt[:, 1],  # x
                                                    self._boundaries[-1].f_mask_sequential_gt[:, 2]])  # y
                    f_collided_gt_opposite = deepcopy(self.f[self._boundaries[-1].opposite_tensor[
                                                                 self._boundaries[-1].f_mask_sequential_gt[:,
                                                                 0]],  # q
                                                             self._boundaries[-1].f_mask_sequential_gt[:, 1],  # x
                                                             self._boundaries[-1].f_mask_sequential_gt[:, 2]])  # y
                if self.lattice.D == 3:
                    f_collided_lt = deepcopy(self.f[self._boundaries[-1].f_mask_sequential_lt[:, 0],  # q
                                                    self._boundaries[-1].f_mask_sequential_lt[:, 1],  # x
                                                    self._boundaries[-1].f_mask_sequential_lt[:, 2],  # y
                                                    self._boundaries[-1].f_mask_sequential_lt[:, 3]])  # z
                    f_collided_lt_opposite = deepcopy(self.f[self._boundaries[-1].opposite_tensor[
                                                                 self._boundaries[-1].f_mask_sequential_lt[:,
                                                                 0]],  # q
                                                             self._boundaries[-1].f_mask_sequential_lt[:, 1],  # x
                                                             self._boundaries[-1].f_mask_sequential_lt[:, 2],  # y
                                                             self._boundaries[-1].f_mask_sequential_lt[:, 3]])  # z

                    f_collided_gt = deepcopy(self.f[self._boundaries[-1].f_mask_sequential_gt[:, 0],  # q
                                                    self._boundaries[-1].f_mask_sequential_gt[:, 1],  # x
                                                    self._boundaries[-1].f_mask_sequential_gt[:, 2],  # y
                                                    self._boundaries[-1].f_mask_sequential_gt[:, 3]])  # z
                    f_collided_gt_opposite = deepcopy(self.f[self._boundaries[-1].opposite_tensor[
                                                                 self._boundaries[-1].f_mask_sequential_gt[:,
                                                                 0]],  # q
                                                             self._boundaries[-1].f_mask_sequential_gt[:, 1],  # x
                                                             self._boundaries[-1].f_mask_sequential_gt[:, 2],  # y
                                                             self._boundaries[-1].f_mask_sequential_gt[:, 3]])  # z

                # f_collided in compact storage-format (because torch.to_sparse() doesn't allow value-assignment and/or batch-indexing)
                self.f_collided_lt = torch.stack((f_collided_lt, f_collided_lt_opposite), dim=1)
                self.f_collided_gt = torch.stack((f_collided_gt, f_collided_gt_opposite), dim=1)
                # print("f_collided_lt.shape:", self.f_collided_lt.shape)
                # print("f_mask_sequential_lt.shape", self._boundaries[-1].f_mask_sequential_lt.shape)
        log_VRAM("SIM_init (f_collided subtensors assigned/stacked)")

    def step(self, num_steps):
        """ Take num_steps stream-and-collision steps and return performance in MLUPS.
        M.Bille: added force_calculation on object/boundaries
        M.Bille: added halfway bounce back boundary
        """
        start = timer()
        log_VRAM("SIM_step (start)")
        if self.i == 0:  # if this is the first timestep, calc. initial force on Object/walls/boundary/obstacle and call reporters
            # reporters are called before the first timestep
            self._report()
        for _ in range(num_steps):  # simulate num_step timesteps
            time1 = timer()
            ### COLLISION
            # Perform the collision routine everywhere, expect where the no_collision_mask is true
            # ...and store post-collision population for halfway-bounce-back boundary condition
            log_VRAM("SIM_step (for loop)")
            self.f = torch.where(self.no_collision_mask, self.f, self.collision(self.f))
            log_VRAM("SIM_step (after collision)")
            if self.store_f_collided:
                if self.lattice.D == 2:
                    f_collided_lt = torch.clone(self.f[self._boundaries[-1].f_mask_sequential_lt[:, 0],  # q
                                                       self._boundaries[-1].f_mask_sequential_lt[:, 1],  # x
                                                       self._boundaries[-1].f_mask_sequential_lt[:, 2]])  # y
                    f_collided_lt_opposite = torch.clone(self.f[self._boundaries[-1].opposite_tensor[
                                                                    self._boundaries[-1].f_mask_sequential_lt[:,
                                                                    0]],  # q
                                                                self._boundaries[-1].f_mask_sequential_lt[:, 1],  # x
                                                                self._boundaries[-1].f_mask_sequential_lt[:, 2]])  # y

                    f_collided_gt = torch.clone(self.f[self._boundaries[-1].f_mask_sequential_gt[:, 0],  # q
                                                       self._boundaries[-1].f_mask_sequential_gt[:, 1],  # x
                                                       self._boundaries[-1].f_mask_sequential_gt[:, 2]])  # y
                    f_collided_gt_opposite = torch.clone(self.f[self._boundaries[-1].opposite_tensor[
                                                                    self._boundaries[-1].f_mask_sequential_gt[:,
                                                                    0]],  # q
                                                                self._boundaries[-1].f_mask_sequential_gt[:, 1],  # x
                                                                self._boundaries[-1].f_mask_sequential_gt[:, 2]])  # y
                if self.lattice.D == 3:
                    f_collided_lt = torch.clone(self.f[self._boundaries[-1].f_mask_sequential_lt[:, 0],  # q
                                                       self._boundaries[-1].f_mask_sequential_lt[:, 1],  # x
                                                       self._boundaries[-1].f_mask_sequential_lt[:, 2],  # y
                                                       self._boundaries[-1].f_mask_sequential_lt[:, 3]])  # z
                    f_collided_lt_opposite = torch.clone(self.f[self._boundaries[-1].opposite_tensor[
                                                                    self._boundaries[-1].f_mask_sequential_lt[:,
                                                                    0]],  # q
                                                                self._boundaries[-1].f_mask_sequential_lt[:, 1],  # x
                                                                self._boundaries[-1].f_mask_sequential_lt[:, 2],  # y
                                                                self._boundaries[-1].f_mask_sequential_lt[:, 3]])  # z

                    f_collided_gt = torch.clone(self.f[self._boundaries[-1].f_mask_sequential_gt[:, 0],  # q
                                                       self._boundaries[-1].f_mask_sequential_gt[:, 1],  # x
                                                       self._boundaries[-1].f_mask_sequential_gt[:, 2],  # y
                                                       self._boundaries[-1].f_mask_sequential_gt[:, 3]])  # z
                    f_collided_gt_opposite = torch.clone(self.f[self._boundaries[-1].opposite_tensor[
                                                                    self._boundaries[-1].f_mask_sequential_gt[:,
                                                                    0]],  # q
                                                                self._boundaries[-1].f_mask_sequential_gt[:, 1],  # x
                                                                self._boundaries[-1].f_mask_sequential_gt[:, 2],  # y
                                                                self._boundaries[-1].f_mask_sequential_gt[:, 3]])  # z

                # log_VRAM("SIM_step (f_collided subtensors assigned)")
                # f_collided in compact storage-format (because torch.to_sparse() doesn't allow value-assignment and/or batch-indexing)
                self.f_collided_lt = torch.stack((f_collided_lt, f_collided_lt_opposite), dim=1)
                self.f_collided_gt = torch.stack((f_collided_gt, f_collided_gt_opposite), dim=1)

            log_VRAM("SIM_step (f_collided subtensors assigned/stacked)")

            time2 = timer()
            ### STREAMING
            self.f = self.streaming(self.f)
            log_VRAM("SIM_step (after streaming)")

            time3 = timer()
            ### BOUNDARY
            # apply boundary conditions
            for boundary in self._boundaries:
                if boundary is not None:
                    if isinstance(boundary,
                                  HalfwayBounceBackBoundary):  # or isinstance(boundary, InterpolatedBounceBackBoundary):
                        self.f = boundary(self.f,
                                          self.f_collided)  # HalfwayBounceBackBoundary needs post-collision_pre-streaming f on boundary nodes to perform reflection of populations within the same timestep
                    elif isinstance(boundary, InterpolatedBounceBackBoundary):
                        self.f = boundary(self.f, self.f_collided_lt, self.f_collided_gt)
                    else:
                        self.f = boundary(self.f)  # all non-HalfwayBounceBackBoundary-BoundaryConditions
                log_VRAM("SIM_step (boundary loop...)")

            # count step
            self.i += 1

            time4 = timer()
            # call reporters
            self._report()
            log_VRAM("SIM_step (after report)")

            time5 = timer()
            self.times[0].append(time2 - time1)  # time to collide
            self.times[1].append(time3 - time2)  # time to stream
            self.times[2].append(time4 - time3)  # time to boundary
            self.times[3].append(time5 - time4)  # time to report
        end = timer()

        # calculate individual runtimes (M.Bille)
        if num_steps > 0:
            self.time_avg = dict(time_collision=sum(self.times[0]) / len(self.times[0]),
                                 time_streaming=sum(self.times[1]) / len(self.times[1]),
                                 time_boundary=sum(self.times[2]) / len(self.times[2]),
                                 time_reporter=sum(self.times[3]) / len(self.times[3]))
        else:
            self.time_avg = dict(time_collision=-1, time_streaming=-1, time_boundary=-1, time_reporter=-1)

        # calculate runtime and performance in MLUPS
        seconds = end - start
        num_grid_points = self.lattice.rho(self.f).numel()
        mlups = num_steps * num_grid_points / 1e6 / seconds
        log_VRAM("SIM_step (end)")
        return mlups

    def _report(self):
        for reporter in self.reporters:
            reporter(self.i, self.flow.units.convert_time_to_pu(self.i), self.f)