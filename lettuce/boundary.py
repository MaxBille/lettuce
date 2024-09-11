"""
Boundary Conditions.

The `__call__` function of a boundary defines its application to the distribution functions.

Boundary conditions can define a mask (a boolean numpy array)
that specifies the grid points on which the boundary
condition operates.

Boundary classes can define two functions `make_no_stream_mask` and `make_no_collision_mask`
that prevent streaming and collisions on the boundary nodes.

The no-stream mask has the same dimensions as the distribution functions (Q, x, y, (z)) .
The no-collision mask has the same dimensions as the grid (x, y, (z)).

"""

# To Do:
#  - the inits for Halfway and Fullway Bounce Back with force calculation (neighbor search) can be outsourced to a function taking mask and lattice and returning tensor(f_mask)
#  - same for the calc_force_on_boundary method
#  - fullway and halfway bounce back could be fitted into one class and specified by parameter (hw/fw) determining the use of how call acts and if no_stream is used (hw)

import torch
import numpy as np
import time

# from build.lib.lettuce.boundary import InterpolatedBounceBackBoundary
from lettuce import LettuceException
from lettuce.lattices import Lattice

__all__ = ["BounceBackBoundary", "HalfwayBounceBackBoundary", "FullwayBounceBackBoundary",
           "AntiBounceBackOutlet", "EquilibriumBoundaryPU", "EquilibriumOutletP", "SlipBoundary",
           "InterpolatedBounceBackBoundary", "InterpolatedBounceBackBoundary_compact_v1", "InterpolatedBounceBackBoundary_compact_v2", "InterpolatedBounceBackBoundary_occ",
           "FullwayBounceBackBoundary_compact", "HalfwayBounceBackBoundary_compact_v1", "HalfwayBounceBackBoundary_compact_v2",
           "HalfwayBounceBackBoundary_compact_v3", "PartiallySaturatedBoundary"]


class PartiallySaturatedBoundary:
    """
    Partially saturated boundary condition using a partial combination of standard full-way bounce back and
    BGK-Collision, first presented by Noble and Torczynski (1998), see Krüger et al., pp. 448.
    """

    # this may be just as efficient as a compact version, b/c the boundary is actually used on all nodes even within the object
    def __init__(self, mask: torch.Tensor, lattice: Lattice, tau: float, saturation: float):
        self.mask = mask
        self.lattice = lattice
        self.tau = tau
        self.B = saturation * (tau - 0.5) / ((1 - saturation) + (tau - 0.5))  # B(epsilon, theta), Krüger p. 448ff
        return

    def __call__(self, f):
        rho = self.lattice.rho(f)
        u = self.lattice.u(f, rho=rho)
        feq = self.lattice.equilibrium(rho, u)
        # TODO: benchmark and possibly use indices (like _compact)
        #  and/or calculate feq twice within torch.where (like _less_memory)
        f = torch.where(self.mask, f - (1.0 - self.B) / self.tau * (f - feq)
                        + self.B * ((f[self.lattice.stencil.opposite] - feq[self.lattice.stencil.opposite])
                                    - (f - self.lattice.equilibrium(rho, torch.zeros_like(u)))), f)
        return f

    def make_no_collision_mask(self, f_shape):
        assert self.mask.shape == f_shape[1:]
        return self.mask


class SolidBoundaryData(dict):
    f_index_lt: np.ndarray
    f_index_gt: np.ndarray
    d_lt: np.ndarray
    d_gt: np.ndarray
    points_inside: np.ndarray
    solid_mask: np.ndarray
    not_intersected: np.ndarray = np.ndarray([])


class InterpolatedBounceBackBoundary:
    """Interpolated Bounce Back Boundary Condition first introduced by Bouzidi et al. (2001), as described in Kruger et al.
        (2017)
        - improvement of the simple bounce back (SBB) Algorithm, used in Fullway and/or Halfway Boucne Back (FWBB, HWBB)
        Boundary Conditions (see FullwayBounceBackBoundary and HalfwayBounceBackBoundary classes)
        - linear or quadratic interpolation of populations to retain the true boundary location between fluid- and
        solid-node
        
        * version 1.0: interpolation of a cylinder (circular in xy, axis along z). Axis position x_center, y_center with
        radius (ALL in LU!)
        NOTE: a mathematical condition for the boundary surface has to be known for calculation of the intersection point
        of boundary link and boundary surface for interpolation!
    """

    def __init__(self, mask, lattice, x_center, y_center, radius, interpolation_order=1):
        t_init_start = time.time()
        self.interpolation_order = interpolation_order
        self.mask = mask  # location of solid-nodes
        self.lattice = lattice
        self.force_sum = torch.zeros_like(self.lattice.convert_to_tensor(
            self.lattice.stencil.e[0]))  # summed force vector on all boundary nodes, in D dimensions (x,y,(z))
        ### create f_mask, needed for force-calculation
        # ...(marks all fs which point from fluid to solid (boundary) and considered for momentum exchange)
        if self.lattice.D == 2:
            nx, ny = self.mask.shape  # domain size in x and y
            self.f_mask = np.zeros((self.lattice.Q, nx, ny), dtype=bool)
            # f_mask: [q, nx, ny], marks all fs which point from fluid to solid (boundary)
            #            self.force = np.zeros((nx, ny, 2))  # force in x and y on all individual nodes
            self.d = np.zeros_like(self.f_mask, dtype=float)  # d: [q,x,y] store the link-length per boundary-cutting link
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
                        if not self.mask[a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                         b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny]:
                            # if the neighbour of p is False in the boundary.mask, p is a solid node, neighbouring a fluid node:
                            # ...the direction pointing from the fluid neighbour to solid p is marked on the neighbour
                            self.f_mask[self.lattice.stencil.opposite[i],
                                        a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                        b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny] = 1
                            # f_mask[q,x,y]

                            # calculate intersection point of boundary surface and link ->
                            # ...calculate distance between fluid node and boundary surface on the link
                            px = a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx  # fluid node x-coordinate
                            py = b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny  # fluid node y-coordinate
                            cx = self.lattice.stencil.e[self.lattice.stencil.opposite[i], 0]  # link-direction x to solid node
                            cy = self.lattice.stencil.e[self.lattice.stencil.opposite[i], 1]  # link-direction y to solid node

                            # pq-formula
                            h1 = (px * cx + py * cy - cx * x_center - cy * y_center) / (cx * cx + cy * cy)  # p/2
                            h2 = (px * px + py * py + x_center * x_center + y_center * y_center
                                  - 2 * px * x_center - 2 * py * y_center - radius * radius) / (cx * cx + cy * cy)  # q

                            d1 = - h1 + np.sqrt(h1 * h1 - h2)
                            d2 = - h1 - np.sqrt(h1 * h1 - h2)

                            #print("xb,yb,i,d1,d2 xf, yf, cx, cy:", a[p], b[p],i,d1,d2,px,py,cx,cy)
                            # distance from fluid node to the "true" boundary location
                            if d1 <= 1 and np.isreal(d1):  # d should be between 0 and 1
                                self.d[self.lattice.stencil.opposite[i],
                                       a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                       b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny] = d1
                            elif d2 <= 1 and np.isreal(d2):
                                self.d[self.lattice.stencil.opposite[i],
                                       a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                       b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny] = d2
                            else:
                                print("IBB WARNING: d1 is", d1,"; d2 is", d2, "for boundaryPoint x,y,ci", a[p],b[p],self.lattice.stencil.e[i, 0],self.lattice.stencil.e[i, 1])
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there
        if self.lattice.D == 3:  # like 2D, but in 3D...guess what...
            nx, ny, nz = self.mask.shape
            self.f_mask = np.zeros((self.lattice.Q, nx, ny, nz), dtype=bool)
            #            self.force = np.zeros((nx, ny, nz, 3))
            self.d = np.zeros_like(self.f_mask, dtype=float)  # d: [q,x,y] store the link-length per boundary-cutting link
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
                        if not self.mask[a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                         b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                         c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz]:
                            self.f_mask[self.lattice.stencil.opposite[i],
                                        a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                        b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                        c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz] = 1

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
                                  - 2 * px * x_center - 2 * py * y_center - radius * radius) / (cx * cx + cy * cy)  # q

                            d1 = - h1 + np.sqrt(h1 * h1 - h2)
                            d2 = - h1 - np.sqrt(h1 * h1 - h2)

                            #print("xb,yb,i,d1,d2 xf, yf, cx, cy:", a[p], b[p], i, d1, d2, px, py, cx, cy)
                            # distance from fluid node to the "true" boundary location
                            if d1 <= 1 and np.isreal(d1):  # d should be between 0 and 1
                                self.d[self.lattice.stencil.opposite[i],
                                       a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                       b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                       c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz] = d1
                            elif d2 <= 1 and np.isreal(d2):
                                self.d[self.lattice.stencil.opposite[i],
                                       a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                       b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                       c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz] = d2
                            else:
                                print("IBB WARNING: d1 is", d1,"; d2 is", d2, "for boundaryPoint x,y,ci", a[p],b[p],c[p],self.lattice.stencil.e[i, 0],self.lattice.stencil.e[i, 1],self.lattice.stencil.e[i, 2])
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there
        self.f_mask = self.lattice.convert_to_tensor(self.f_mask)
        self.d = self.lattice.convert_to_tensor(self.d)
        self.f_collided = torch.zeros_like(self.f_mask, device=self.lattice.device, dtype=self.lattice.dtype)
        print("IBB initialization took " + str(time.time() - t_init_start) + "seconds")

    def __call__(self, f):

        if self.interpolation_order == 2:
            print("warning: not implemented")
        else:  # interpolation_order==1:
            # f_tmp = f_collided[i,x_b]_interpolation before bounce
            f_tmp = torch.where(self.d <= 0.5,  # if d<=1/2
                                2 * self.d * self.f_collided + (1 - 2 * self.d) * f, # interpolate from second fluid node
                                (1 / (2 * self.d)) * self.f_collided + (1 - 1 / (2 * self.d)) * self.f_collided[self.lattice.stencil.opposite])  # else: interpolate from opposing populations on x_b
            # (?) 1-1/(2d) ODER (2d-1)/2d, welches ist numerisch exakter?
            # f_collided an x_f entspricht f_streamed an x_b, weil entlang des links ohne collision gestreamt wird!
            # ... d.h. f_collided[i,x_f] entspricht f[i,x_b]
            f = torch.where(self.f_mask[self.lattice.stencil.opposite], f_tmp[self.lattice.stencil.opposite], f)
            # HWBB: f = torch.where(self.f_mask[self.lattice.stencil.opposite], f_collided[self.lattice.stencil.opposite], f)

        self.calc_force_on_boundary(f)
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

    def calc_force_on_boundary(self, f_bounced):
        # momentum exchange according to Bouzidi et al. (2001), equation 11.8 in Kruger et al. (2017) p.445 // watch out for wrong signs. Kruger states "-", but "+" gives correct result
        # ...original paper (Bouzidi 2001) and Ginzburg followup (2003) state a "+" as well...
        #OLDtmp = torch.where(self.f_mask, f_collided, torch.zeros_like(f_bounced)) \
        #OLD      - torch.where(self.f_mask, f_bounced[self.lattice.stencil.opposite], torch.zeros_like(f_bounced))
        #WRONG: tmp = torch.where(self.f_mask, f_collided - f_bounced[self.lattice.stencil.opposite], torch.zeros_like(f_bounced)) #WRONG
        tmp = torch.where(self.f_mask, self.f_collided + f_bounced[self.lattice.stencil.opposite], torch.zeros_like(f_bounced))  #RIGHT
        self.force_sum = torch.einsum('i..., id -> d', tmp, self.lattice.e)  # CALCULATE FORCE / v3.0 - M.Bille: dx_lu = dt_lu is allways 1 (!)

    def store_f_collided(self, f_collided):
        self.f_collided = torch.clone(f_collided)


class InterpolatedBounceBackBoundary_compact_v1:

    def __init__(self, mask, lattice, x_center, y_center, radius, interpolation_order=1):
        t_init_start = time.time()
        self.interpolation_order = interpolation_order
        self.mask = mask  # location of solid-nodes
        self.lattice = lattice
        self.force_sum = torch.zeros_like(self.lattice.convert_to_tensor(self.lattice.stencil.e[0]))  # summed force vector on all boundary nodes, in D dimensions (x,y,(z))

        self.f_index_lt = []  # indices of relevant populations (for bounce back and force-calculation) with d<=0.5
        self.f_index_gt = []  # indices of relevant populations (for bounce back and force-calculation) with d>0.5
        self.d_lt = []  # distances between node and boundary for d<0.5
        self.d_gt = []  # distances between node and boundary for d>0.5

        # searching boundary-fluid-interface and append indices to f_index, distance to boundary to d
        if self.lattice.D == 2:
            nx, ny = mask.shape  # domain size in x and y
            self.f_mask = np.zeros((self.lattice.Q, nx, ny), dtype=bool)  # f_mask: [q, nx, ny], marks all fs which point from fluid to solid (boundary), needed to collect f_collided in simulation
            a, b = np.where(mask)  # x- and y-index of boundaryTRUE nodes for iteration over boundary area

            for p in range(0, len(a)):  # for all TRUE-nodes in boundary.mask
                for i in range(0, self.lattice.Q):  # for all stencil-directions c_i (lattice.stencil.e in lettuce)
                    # check for boundary-nodes neighboring the domain-border.
                    # ...they have to take the periodicity into account...
                    border = np.zeros(self.lattice.D, dtype=int)

                    if a[p] == 0 and self.lattice.stencil.e[i, 0] == -1:  # searching border on left [x]
                        border[0] = -1
                    elif a[p] == nx - 1 and self.lattice.e[i, 0] == 1:  # searching border on right [x]
                        border[0] = 1

                    if b[p] == 0 and self.lattice.stencil.e[i, 1] == -1:  # searching border on left [y]
                        border[1] = -1
                    elif b[p] == ny - 1 and self.lattice.e[i, 1] == 1:  # searching border on right [y]
                        border[1] = 1

                    try:  # try in case the neighboring cell does not exist (= an f pointing out of the simulation domain)
                        if not mask[a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                    b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny]:
                            # if the neighbour of p is False in the boundary.mask, p is a solid node, neighbouring a fluid node:
                            # ...the direction pointing from the fluid neighbour to solid p is marked on the neighbour

                            self.f_mask[self.lattice.stencil.opposite[i],
                                        a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                        b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny] = 1  # f_mask[q,x,y], marks all fs which point from fluid to solid (boundary)

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

                            # distance from fluid node to the "true" boundary location
                            # choose correct d and assign d and f_index
                            if d1 <= 1 and np.isreal(d1):  # d should be between 0 and 1

                                if d1 <= 0.5:
                                    self.d_lt.append(d1)
                                    self.f_index_lt.append([self.lattice.stencil.opposite[i],
                                                            a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                            b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny])
                                else:  # d>0.5
                                    self.d_gt.append(d1)
                                    self.f_index_gt.append([self.lattice.stencil.opposite[i],
                                                            a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                            b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny])

                            elif d2 <= 1 and np.isreal(d2):  # d should be between 0 and 1

                                if d2 <= 0.5:
                                    self.d_lt.append(d2)
                                    self.f_index_lt.append([self.lattice.stencil.opposite[i],
                                                            a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                            b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny])
                                else:  # d>0.5
                                    self.d_gt.append(d2)
                                    self.f_index_gt.append([self.lattice.stencil.opposite[i],
                                                            a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                            b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny])
                            else:  # neither d1 or d2 is real and between 0 and 1
                                print("IBB WARNING: d1 is", d1, "; d2 is", d2, "for boundaryPoint x,y,ci", a[p],
                                      b[p], self.lattice.stencil.e[i, 0], self.lattice.stencil.e[i, 1],
                                      self.lattice.stencil.e[i, 2])
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there

        if self.lattice.D == 3:  # like 2D, but in 3D...guess what...
            nx, ny, nz = mask.shape
            self.f_mask = np.zeros((self.lattice.Q, nx, ny, nz), dtype=bool)
            a, b, c = np.where(mask)

            for p in range(0, len(a)):
                for i in range(0, self.lattice.Q):
                    border = np.zeros(self.lattice.D, dtype=int)
                    # x - direction
                    if a[p] == 0 and self.lattice.stencil.e[i, 0] == -1:  # searching border on left
                        border[0] = -1
                    elif a[p] == nx - 1 and self.lattice.e[i, 0] == 1:  # searching border on right
                        border[0] = 1
                    # y - direction
                    if b[p] == 0 and self.lattice.stencil.e[i, 1] == -1:  # searching border on left
                        border[1] = -1
                    elif b[p] == ny - 1 and self.lattice.e[i, 1] == 1:  # searching border on right
                        border[1] = 1
                    # z - direction
                    if c[p] == 0 and self.lattice.stencil.e[i, 2] == -1:  # searching border on left
                        border[2] = -1
                    elif c[p] == nz - 1 and self.lattice.e[i, 2] == 1:  # searching border on right
                        border[2] = 1

                    try:  # try in case the neighboring cell does not exist (an f pointing out of simulation domain)
                        if not mask[a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                    b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                    c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz]:
                            self.f_mask[self.lattice.stencil.opposite[i],
                                        a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                        b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                        c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz] = 1

                            # calculate intersection point of boundary surface and link ->
                            # ...calculate distance between fluid node and boundary surface on the link
                            px = a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx  # fluid node x-coordinate
                            py = b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny  # fluid node y-coordinate
                            # Z-coodinate not needed for cylinder !

                            cx = self.lattice.stencil.e[
                                self.lattice.stencil.opposite[i], 0]  # link-direction x to solid node
                            cy = self.lattice.stencil.e[
                                self.lattice.stencil.opposite[i], 1]  # link-direction y to solid node
                            # Z-coodinate not needed for cylinder !

                            # pq-formula
                            h1 = (px * cx + py * cy - cx * x_center - cy * y_center) / (cx * cx + cy * cy)  # p/2
                            h2 = (px * px + py * py + x_center * x_center + y_center * y_center
                                  - 2 * px * x_center - 2 * py * y_center - radius * radius) / (
                                         cx * cx + cy * cy)  # q

                            d1 = - h1 + np.sqrt(h1 * h1 - h2)
                            d2 = - h1 - np.sqrt(h1 * h1 - h2)

                            # print("xb,yb,i,d1,d2 xf, yf, cx, cy:", a[p], b[p], i, d1, d2, px, py, cx, cy)

                            # distance from fluid node to the "true" boundary location
                            # choose correct d and assign d and f_index
                            if d1 <= 1 and np.isreal(d1):  # d should be between 0 and 1

                                if d1 <= 0.5:
                                    self.d_lt.append(d1)
                                    self.f_index_lt.append([self.lattice.stencil.opposite[i],
                                                              a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                              b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                                              c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz])
                                else:  # d>0.5
                                    self.d_gt.append(d1)
                                    self.f_index_gt.append([self.lattice.stencil.opposite[i],
                                                              a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                              b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                                              c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz])

                            elif d2 <= 1 and np.isreal(d2): # d should be between 0 and 1

                                if d2 <= 0.5:
                                    self.d_lt.append(d2)
                                    self.f_index_lt.append([self.lattice.stencil.opposite[i],
                                                              a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                              b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                                              c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz])
                                else:  # d>0.5
                                    self.d_gt.append(d2)
                                    self.f_index_gt.append([self.lattice.stencil.opposite[i],
                                                              a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                              b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                                              c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz])
                            else:  # neither d1 or d2 is real and between 0 and 1
                                print("IBB WARNING: d1 is", d1, "; d2 is", d2, "for boundaryPoint x,y,z,ci", a[p],
                                      b[p], c[p], self.lattice.stencil.e[i, 0], self.lattice.stencil.e[i, 1],
                                      self.lattice.stencil.e[i, 2])
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there

        # convert relevant tensors:
        self.f_mask = self.lattice.convert_to_tensor(self.f_mask)
        self.f_index_lt = torch.tensor(np.array(self.f_index_lt), device=self.lattice.device, dtype=torch.int64)  # the batch-index has to be integer
        self.f_index_gt = torch.tensor(np.array(self.f_index_gt), device=self.lattice.device, dtype=torch.int64)  # the batch-index has to be integer
        self.d_lt = self.lattice.convert_to_tensor(np.array(self.d_lt))
        self.d_gt = self.lattice.convert_to_tensor(np.array(self.d_gt))
        self.opposite_tensor = torch.tensor(self.lattice.stencil.opposite, device=self.lattice.device, dtype=torch.int64)  # batch-index has to be a tensor

        self.f_collided = []

        if lattice.D == 2:
            fc_q, fc_x, fc_y = torch.where(self.f_mask + self.f_mask[self.lattice.stencil.opposite])  # which populations of f_collided to store
            self.fc_index = torch.stack((fc_q, fc_x, fc_y))
        if lattice.D == 3:
            fc_q, fc_x, fc_y, fc_z = torch.where(self.f_mask + self.f_mask[self.lattice.stencil.opposite])  # which populations of f_collided to store
            self.fc_index = torch.stack((fc_q, fc_x, fc_y, fc_z))

        print("IBB initialization took " + str(time.time() - t_init_start) + "seconds")

    def __call__(self, f):

        # BOUNCE (Bouzidi et al. (2001), as described in Kruger et al. (2017))
        if self.lattice.D == 2:
            # if d <= 0.5
            f[self.opposite_tensor[self.f_index_lt[:, 0]],
              self.f_index_lt[:, 1],
              self.f_index_lt[:, 2]] = 2 * self.d_lt * self.f_collided.to_dense()[self.f_index_lt[:, 0],
                                                                             self.f_index_lt[:, 1],
                                                                             self.f_index_lt[:, 2]] \
                                       + (1 - 2 * self.d_lt) * f[self.f_index_lt[:, 0],
                                                                 self.f_index_lt[:, 1],
                                                                 self.f_index_lt[:, 2]]
            # if d > 0.5
            f[self.opposite_tensor[self.f_index_gt[:, 0]],
              self.f_index_gt[:, 1],
              self.f_index_gt[:, 2]] = (1 / (2 * self.d_gt)) * self.f_collided.to_dense()[self.f_index_gt[:, 0],
                                                                                     self.f_index_gt[:, 1],
                                                                                     self.f_index_gt[:, 2]] \
                                       + (1 - 1 / (2 * self.d_gt)) * self.f_collided.to_dense()[
                                           self.opposite_tensor[self.f_index_gt[:, 0]],
                                           self.f_index_gt[:, 1],
                                           self.f_index_gt[:, 2]]
        if self.lattice.D == 3:
            # if d <= 0.5
            f[self.opposite_tensor[self.f_index_lt[:, 0]],
              self.f_index_lt[:, 1],
              self.f_index_lt[:, 2],
              self.f_index_lt[:, 3]] = 2 * self.d_lt * self.f_collided.to_dense()[self.f_index_lt[:, 0],
                                                                             self.f_index_lt[:, 1],
                                                                             self.f_index_lt[:, 2],
                                                                             self.f_index_lt[:, 3]] \
                                         + (1 - 2 * self.d_lt) * f[self.f_index_lt[:, 0],
                                                                   self.f_index_lt[:, 1],
                                                                   self.f_index_lt[:, 2],
                                                                   self.f_index_lt[:, 3]]
            # if d > 0.5
            f[self.opposite_tensor[self.f_index_gt[:, 0]],
              self.f_index_gt[:, 1],
              self.f_index_gt[:, 2],
              self.f_index_gt[:, 3]] = (1 / (2 * self.d_gt)) * self.f_collided.to_dense()[self.f_index_gt[:, 0],
                                                                                     self.f_index_gt[:, 1],
                                                                                     self.f_index_gt[:, 2],
                                                                                     self.f_index_gt[:, 3]] \
                                         + (1 - 1 / (2 * self.d_gt)) * self.f_collided.to_dense()[
                                                     self.opposite_tensor[self.f_index_gt[:, 0]],
                                                     self.f_index_gt[:, 1],
                                                     self.f_index_gt[:, 2],
                                                     self.f_index_gt[:, 3]]

        # CALCULATE FORCE
        self.calc_force_on_boundary(f)
        return f

    def make_no_stream_mask(self, f_shape):
        assert self.mask.shape == f_shape[1:]  # all dimensions of f except the 0th (q)
        # no_stream_mask has to be dimensions: (q,x,y,z) (z optional), but CAN be (x,y,z) (z optional).
        # ...in the latter case, torch.where broadcasts the mask to (q,x,y,z), so ALL q populations of a lattice-node are marked equally
        return self.lattice.convert_to_tensor(self.mask)

    def make_no_collision_mask(self, f_shape):
        # INFO: pay attention to the initialization of observable/moment-fields (u, rho,...) on the boundary nodes,
        # ...in the initial solution of your flow, especially if visualization or post processing uses the field-values
        # ...in the whole domain (including the boundary region)!
        assert self.mask.shape == f_shape[1:]
        return self.lattice.convert_to_tensor(self.mask)

    def calc_force_on_boundary(self, f_bounced):
        ### force = e * (f_collided + f_bounced[opposite])
        if self.lattice.D == 2:
            self.force_sum = torch.einsum('i..., id -> d', self.f_collided.to_dense()[self.f_index_lt[:, 0],
                                                                                      self.f_index_lt[:, 1],
                                                                                      self.f_index_lt[:, 2]] \
                                          + f_bounced[self.opposite_tensor[self.f_index_lt[:, 0]],
                                                      self.f_index_lt[:, 1],
                                                      self.f_index_lt[:, 2]],
                                          self.lattice.e[self.f_index_lt[:, 0]]) \
                             + torch.einsum('i..., id -> d', self.f_collided.to_dense()[self.f_index_gt[:, 0],
                                                                                        self.f_index_gt[:, 1],
                                                                                        self.f_index_gt[:, 2]] \
                                            + f_bounced[self.opposite_tensor[self.f_index_gt[:, 0]],
                                                        self.f_index_gt[:, 1],
                                                        self.f_index_gt[:, 2]],
                                            self.lattice.e[self.f_index_gt[:, 0]])
        if self.lattice.D == 3:
            self.force_sum = torch.einsum('i..., id -> d', self.f_collided.to_dense()[self.f_index_lt[:, 0],
                                                                                      self.f_index_lt[:, 1],
                                                                                      self.f_index_lt[:, 2],
                                                                                      self.f_index_lt[:, 3]] \
                                          + f_bounced[self.opposite_tensor[self.f_index_lt[:, 0]],
                                                      self.f_index_lt[:, 1],
                                                      self.f_index_lt[:, 2],
                                                      self.f_index_lt[:, 3]],
                                          self.lattice.e[self.f_index_lt[:, 0]]) \
                             + torch.einsum('i..., id -> d', self.f_collided.to_dense()[self.f_index_gt[:, 0],
                                                                                        self.f_index_gt[:, 1],
                                                                                        self.f_index_gt[:, 2],
                                                                                        self.f_index_gt[:, 3]] \
                                            + f_bounced[self.opposite_tensor[self.f_index_gt[:, 0]],
                                                        self.f_index_gt[:, 1],
                                                        self.f_index_gt[:, 2],
                                                        self.f_index_gt[:, 3]],
                                            self.lattice.e[self.f_index_gt[:, 0]])

    def store_f_collided(self, f_collided):
        if self.lattice.D == 2:
            self.f_collided = torch.clone(torch.sparse_coo_tensor(indices=self.fc_index,
                                                                  values=f_collided[self.fc_index[0], self.fc_index[1],
                                                                                    self.fc_index[2]],
                                                                  size=f_collided.size()))
        if self.lattice.D == 3:
            self.f_collided = torch.clone(torch.sparse_coo_tensor(indices=self.fc_index,
                                                                  values=f_collided[self.fc_index[0], self.fc_index[1],
                                                                                    self.fc_index[2], self.fc_index[3]],
                                                                  size=f_collided.size()))


class InterpolatedBounceBackBoundary_compact_v2:

    def __init__(self, mask, lattice, x_center, y_center, radius, interpolation_order=1):
        t_init_start = time.time()
        self.interpolation_order = interpolation_order
        self.mask = mask  # location of solid-nodes
        self.lattice = lattice
        self.force_sum = torch.zeros_like(self.lattice.convert_to_tensor(
            self.lattice.stencil.e[0]))  # summed force vector on all boundary nodes, in D dimensions (x,y,(z))

        self.f_index_lt = []  # indices of relevant populations (for bounce back and force-calculation) with d<=0.5
        self.f_index_gt = []  # indices of relevant populations (for bounce back and force-calculation) with d>0.5
        self.d_lt = []  # distances between node and boundary for d<0.5
        self.d_gt = []  # distances between node and boundary for d>0.5

        # searching boundary-fluid-interface and append indices to f_index, distance to boundary to d
        if self.lattice.D == 2:
            nx, ny = mask.shape  # domain size in x and y
            a, b = np.where(mask)  # x- and y-index of boundaryTRUE nodes for iteration over boundary area

            for p in range(0, len(a)):  # for all TRUE-nodes in boundary.mask
                for i in range(0, self.lattice.Q):  # for all stencil-directions c_i (lattice.stencil.e in lettuce)
                    # check for boundary-nodes neighboring the domain-border.
                    # ...they have to take the periodicity into account...
                    border = np.zeros(self.lattice.D, dtype=int)

                    if a[p] == 0 and self.lattice.stencil.e[i, 0] == -1:  # searching border on left [x]
                        border[0] = -1
                    elif a[p] == nx - 1 and self.lattice.e[i, 0] == 1:  # searching border on right [x]
                        border[0] = 1

                    if b[p] == 0 and self.lattice.stencil.e[i, 1] == -1:  # searching border on left [y]
                        border[1] = -1
                    elif b[p] == ny - 1 and self.lattice.e[i, 1] == 1:  # searching border on right [y]
                        border[1] = 1

                    try:  # try in case the neighboring cell does not exist (= an f pointing out of the simulation domain)
                        if not mask[a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                    b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny]:
                            # if the neighbour of p is False in the boundary.mask, p is a solid node, neighbouring a fluid node:
                            # ...the direction pointing from the fluid neighbour to solid p is marked on the neighbour

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

                            # distance from fluid node to the "true" boundary location
                            # choose correct d and assign d and f_index
                            if d1 <= 1 and np.isreal(d1):  # d should be between 0 and 1

                                if d1 <= 0.5:
                                    self.d_lt.append(d1)
                                    self.f_index_lt.append([self.lattice.stencil.opposite[i],
                                                            a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                            b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny])
                                else:  # d>0.5
                                    self.d_gt.append(d1)
                                    self.f_index_gt.append([self.lattice.stencil.opposite[i],
                                                            a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                            b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny])

                            elif d2 <= 1 and np.isreal(d2):  # d should be between 0 and 1

                                if d2 <= 0.5:
                                    self.d_lt.append(d2)
                                    self.f_index_lt.append([self.lattice.stencil.opposite[i],
                                                            a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                            b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny])
                                else:  # d>0.5
                                    self.d_gt.append(d2)
                                    self.f_index_gt.append([self.lattice.stencil.opposite[i],
                                                            a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                            b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny])
                            else:  # neither d1 or d2 is real and between 0 and 1
                                print("IBB WARNING: d1 is", d1, "; d2 is", d2, "for boundaryPoint x,y,ci", a[p],
                                      b[p], self.lattice.stencil.e[i, 0], self.lattice.stencil.e[i, 1],
                                      self.lattice.stencil.e[i, 2])
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there

        if self.lattice.D == 3:  # like 2D, but in 3D...guess what...
            nx, ny, nz = mask.shape
            a, b, c = np.where(mask)

            for p in range(0, len(a)):
                for i in range(0, self.lattice.Q):
                    border = np.zeros(self.lattice.D, dtype=int)
                    # x - direction
                    if a[p] == 0 and self.lattice.stencil.e[i, 0] == -1:  # searching border on left
                        border[0] = -1
                    elif a[p] == nx - 1 and self.lattice.e[i, 0] == 1:  # searching border on right
                        border[0] = 1
                    # y - direction
                    if b[p] == 0 and self.lattice.stencil.e[i, 1] == -1:  # searching border on left
                        border[1] = -1
                    elif b[p] == ny - 1 and self.lattice.e[i, 1] == 1:  # searching border on right
                        border[1] = 1
                    # z - direction
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
                            # Z-coodinate not needed for cylinder !

                            cx = self.lattice.stencil.e[
                                self.lattice.stencil.opposite[i], 0]  # link-direction x to solid node
                            cy = self.lattice.stencil.e[
                                self.lattice.stencil.opposite[i], 1]  # link-direction y to solid node
                            # Z-coodinate not needed for cylinder !

                            # pq-formula
                            h1 = (px * cx + py * cy - cx * x_center - cy * y_center) / (cx * cx + cy * cy)  # p/2
                            h2 = (px * px + py * py + x_center * x_center + y_center * y_center
                                  - 2 * px * x_center - 2 * py * y_center - radius * radius) / (
                                         cx * cx + cy * cy)  # q

                            d1 = - h1 + np.sqrt(h1 * h1 - h2)
                            d2 = - h1 - np.sqrt(h1 * h1 - h2)

                            # print("xb,yb,i,d1,d2 xf, yf, cx, cy:", a[p], b[p], i, d1, d2, px, py, cx, cy)

                            # distance from fluid node to the "true" boundary location
                            # choose correct d and assign d and f_index
                            if d1 <= 1 and np.isreal(d1):  # d should be between 0 and 1

                                if d1 <= 0.5:
                                    self.d_lt.append(d1)
                                    self.f_index_lt.append([self.lattice.stencil.opposite[i],
                                                            a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                            b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                                            c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz])
                                else:  # d>0.5
                                    self.d_gt.append(d1)
                                    self.f_index_gt.append([self.lattice.stencil.opposite[i],
                                                            a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                            b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                                            c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz])

                            elif d2 <= 1 and np.isreal(d2):  # d should be between 0 and 1

                                if d2 <= 0.5:
                                    self.d_lt.append(d2)
                                    self.f_index_lt.append([self.lattice.stencil.opposite[i],
                                                            a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                            b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                                            c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz])
                                else:  # d>0.5
                                    self.d_gt.append(d2)
                                    self.f_index_gt.append([self.lattice.stencil.opposite[i],
                                                            a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                            b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                                            c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz])
                            else:  # neither d1 or d2 is real and between 0 and 1
                                print("IBB WARNING: d1 is", d1, "; d2 is", d2, "for boundaryPoint x,y,z,ci", a[p],
                                      b[p], c[p], self.lattice.stencil.e[i, 0], self.lattice.stencil.e[i, 1],
                                      self.lattice.stencil.e[i, 2])
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there

        # convert relevant tensors:
        self.f_index_lt = torch.tensor(np.array(self.f_index_lt), device=self.lattice.device,
                                       dtype=torch.int64)  # the batch-index has to be integer
        self.f_index_gt = torch.tensor(np.array(self.f_index_gt), device=self.lattice.device,
                                       dtype=torch.int64)  # the batch-index has to be integer
        self.d_lt = self.lattice.convert_to_tensor(np.array(self.d_lt))
        self.d_gt = self.lattice.convert_to_tensor(np.array(self.d_gt))
        self.opposite_tensor = torch.tensor(self.lattice.stencil.opposite, device=self.lattice.device,
                                            dtype=torch.int64)  # batch-index has to be a tensor

        f_collided_lt = torch.zeros_like(self.d_lt)  # float-tensor with number of (x_b nodes with d<=0.5) values
        f_collided_gt = torch.zeros_like(self.d_gt)  # float-tensor with number of (x_b nodes with d>0.5) values
        f_collided_lt_opposite = torch.zeros_like(self.d_lt)
        f_collided_gt_opposite = torch.zeros_like(self.d_gt)
        self.f_collided_lt = torch.stack((f_collided_lt, f_collided_lt_opposite), dim=1)
        self.f_collided_gt = torch.stack((f_collided_gt, f_collided_gt_opposite), dim=1)

        print("IBB initialization took " + str(time.time() - t_init_start) + "seconds")

    def __call__(self, f):
        ## f_collided_lt = [f_collided_lt, f_collided_lt.opposite] (!) in compact storage-layout

        if self.lattice.D == 2:
            # BOUNCE
            # if d <= 0.5
            f[self.opposite_tensor[self.f_index_lt[:, 0]],
              self.f_index_lt[:, 1],
              self.f_index_lt[:, 2]] = 2 * self.d_lt * self.f_collided_lt[:, 0] + (1 - 2 * self.d_lt) * f[self.f_index_lt[:, 0],
                                                                                                     self.f_index_lt[:, 1],
                                                                                                     self.f_index_lt[:, 2]]
            # if d > 0.5
            f[self.opposite_tensor[self.f_index_gt[:, 0]],
              self.f_index_gt[:, 1],
              self.f_index_gt[:, 2]] = (1 / (2 * self.d_gt)) * self.f_collided_gt[:, 0] + (1 - 1 / (2 * self.d_gt)) * self.f_collided_gt[:, 1]

        if self.lattice.D == 3:
            # BOUNCE
            # if d <= 0.5
            f[self.opposite_tensor[self.f_index_lt[:, 0]],
              self.f_index_lt[:, 1],
              self.f_index_lt[:, 2],
              self.f_index_lt[:, 3]] = 2 * self.d_lt * self.f_collided_lt[:, 0] + (1 - 2 * self.d_lt) * f[self.f_index_lt[:, 0],
                                                                                                     self.f_index_lt[:, 1],
                                                                                                     self.f_index_lt[:, 2],
                                                                                                     self.f_index_lt[:, 3]]
            # if d > 0.5
            f[self.opposite_tensor[self.f_index_gt[:, 0]],
              self.f_index_gt[:, 1],
              self.f_index_gt[:, 2],
              self.f_index_gt[:, 3]] = (1 / (2 * self.d_gt)) * self.f_collided_gt[:, 0] + (1 - 1 / (2 * self.d_gt)) * self.f_collided_gt[:, 1]


        # CALC. FORCE on boundary (MEM, MEA)
        self.calc_force_on_boundary(f)
        return f

    def make_no_stream_mask(self, f_shape):
        assert self.mask.shape == f_shape[1:]  # all dimensions of f except the 0th (q)
        # no_stream_mask has to be dimensions: (q,x,y,z) (z optional), but CAN be (x,y,z) (z optional).
        # ...in the latter case, torch.where broadcasts the mask to (q,x,y,z), so ALL q populations of a lattice-node are marked equally
        return self.lattice.convert_to_tensor(self.mask)

    def make_no_collision_mask(self, f_shape):
        # INFO: pay attention to the initialization of observable/moment-fields (u, rho,...) on the boundary nodes,
        # ...in the initial solution of your flow, especially if visualization or post processing uses the field-values
        # ...in the whole domain (including the boundary region)!
        assert self.mask.shape == f_shape[1:]
        return self.lattice.convert_to_tensor(self.mask)

    def calc_force_on_boundary(self, f_bounced):
        ### force = e * (f_collided + f_bounced[opp.])
        if self.lattice.D == 2:
            self.force_sum = torch.einsum('i..., id -> d',
                                          self.f_collided_lt[:, 0] + f_bounced[
                                              self.opposite_tensor[self.f_index_lt[:, 0]],
                                              self.f_index_lt[:, 1],
                                              self.f_index_lt[:, 2]],
                                          self.lattice.e[self.f_index_lt[:, 0]]) \
                             + torch.einsum('i..., id -> d',
                                            self.f_collided_gt[:, 0] + f_bounced[
                                                self.opposite_tensor[self.f_index_gt[:, 0]],
                                                self.f_index_gt[:, 1],
                                                self.f_index_gt[:, 2]],
                                            self.lattice.e[self.f_index_gt[:, 0]])
        if self.lattice.D == 3:
            self.force_sum = torch.einsum('i..., id -> d',
                                          self.f_collided_lt[:, 0] + f_bounced[
                                              self.opposite_tensor[self.f_index_lt[:, 0]],
                                              self.f_index_lt[:, 1],
                                              self.f_index_lt[:, 2],
                                              self.f_index_lt[:, 3]],
                                          self.lattice.e[self.f_index_lt[:, 0]]) \
                             + torch.einsum('i..., id -> d',
                                            self.f_collided_gt[:, 0] + f_bounced[
                                                self.opposite_tensor[self.f_index_gt[:, 0]],
                                                self.f_index_gt[:, 1],
                                                self.f_index_gt[:, 2],
                                                self.f_index_gt[:, 3]],
                                            self.lattice.e[self.f_index_gt[:, 0]])

    def store_f_collided(self, f_collided):
        if self.lattice.D == 2:
            self.f_collided_lt[:, 0] = torch.clone(f_collided[self.f_index_lt[:, 0],  # q
                                                          self.f_index_lt[:, 1],  # x
                                                          self.f_index_lt[:, 2]])  # y
            self.f_collided_lt[:, 1] = torch.clone(f_collided[self.opposite_tensor[self.f_index_lt[:,0]],  # q
                                                          self.f_index_lt[:, 1],  # x
                                                          self.f_index_lt[:, 2]])  # y

            self.f_collided_gt[:, 0] = torch.clone(f_collided[self.f_index_gt[:, 0],  # q
                                                          self.f_index_gt[:, 1],  # x
                                                          self.f_index_gt[:, 2]])  # y
            self.f_collided_gt[:, 1] = torch.clone(f_collided[self.opposite_tensor[self.f_index_gt[:,0]],  # q
                                                          self.f_index_gt[:, 1],  # x
                                                          self.f_index_gt[:, 2]])  # y
        if self.lattice.D == 3:
            self.f_collided_lt[:, 0] = torch.clone(f_collided[self.f_index_lt[:, 0],  # q
                                                          self.f_index_lt[:, 1],  # x
                                                          self.f_index_lt[:, 2],  # y
                                                          self.f_index_lt[:, 3]])  # z
            self.f_collided_lt[:, 1] = torch.clone(f_collided[self.opposite_tensor[self.f_index_lt[:,0]],  # q
                                                          self.f_index_lt[:, 1],  # x
                                                          self.f_index_lt[:, 2],  # y
                                                          self.f_index_lt[:, 3]])  # z

            self.f_collided_gt[:, 0] = torch.clone(f_collided[self.f_index_gt[:, 0],  # q
                                                              self.f_index_gt[:, 1],  # x
                                                              self.f_index_gt[:, 2],  # y
                                                              self.f_index_gt[:, 3]])  # z
            self.f_collided_gt[:, 1] = torch.clone(f_collided[self.opposite_tensor[self.f_index_gt[:, 0]],  # q
                                                              self.f_index_gt[:, 1],  # x
                                                              self.f_index_gt[:, 2],  # y
                                                              self.f_index_gt[:, 3]])  # z

class InterpolatedBounceBackBoundary_occ:
    """Interpolated Bounce Back Boundary Condition first introduced by Bouzidi et al. (2001), as described in Kruger et al.
        (2017)
        - linear or quadratic interpolation of populations to retain the true boundary location between fluid- and
        solid-node
        * version 2.0: using given indices and distances between fluid- and solid-node
        of boundary link and boundary surface for interpolation!
    """

    def __init__(self, mask, lattice: Lattice, solid_boundary_data: SolidBoundaryData, calc_force=None, ad_enabled=False):
        t_init_start = time.time()
        if not hasattr(solid_boundary_data, "solid_mask"):
            self.mask = mask
            self.solid_mask = mask
        else:
            self.mask = solid_boundary_data.solid_mask
            self.solid_mask = solid_boundary_data.solid_mask
        self.lattice = lattice
        if calc_force is not None:
            self.force_sum = torch.zeros_like(self.lattice.convert_to_tensor(
                self.lattice.stencil.e[0]))  # summed force vector on all boundary nodes, in D dimensions (x,y,(z))
            self.calc_force = True
        else:
            self.calc_force = False

        # convert relevant tensors:
        ### TODO: fix batch-index-datatype...?
        self.f_index_lt = torch.tensor(solid_boundary_data.f_index_lt, device=self.lattice.device, dtype=torch.int64)  # the batch-index has to be integer
        self.f_index_gt = torch.tensor(solid_boundary_data.f_index_gt, device=self.lattice.device, dtype=torch.int64)  # the batch-index has to be integer

        self.d_lt = lattice.convert_to_tensor(solid_boundary_data.d_lt)
        self.d_gt = lattice.convert_to_tensor(solid_boundary_data.d_gt)
        self.opposite_tensor = torch.tensor(self.lattice.stencil.opposite, device=self.lattice.device, dtype=torch.int64)  # batch-index has to be a tensor

        # f_collided_lt = torch.zeros_like(self.d_lt)  # float-tensor with number of (x_b nodes with d<=0.5) values
        # f_collided_gt = torch.zeros_like(self.d_gt)  # float-tensor with number of (x_b nodes with d>0.5) values
        # f_collided_lt_opposite = torch.zeros_like(self.d_lt)
        # f_collided_gt_opposite = torch.zeros_like(self.d_gt)
        # self.f_collided_lt = torch.stack((f_collided_lt, f_collided_lt_opposite), dim=1)
        # self.f_collided_gt = torch.stack((f_collided_gt, f_collided_gt_opposite), dim=1)
        print(f"IBB initialization took {time.time() - t_init_start:.2f} seconds")

    def __call__(self, f):
        ## f_collided_lt = [f_collided_lt, f_collided_lt.opposite] (!) in compact storage-layout

        if self.lattice.D == 2:
            # BOUNCE
            # if d <= 0.5
            if len(self.f_index_lt) != 0:
                f[self.opposite_tensor[self.f_index_lt[:, 0]],
                self.f_index_lt[:, 1],
                self.f_index_lt[:, 2]] = 2 * self.d_lt * self.f_collided_lt[:, 0] + (1 - 2 * self.d_lt) * f[
                    self.f_index_lt[:, 0],
                    self.f_index_lt[:, 1],
                    self.f_index_lt[:, 2]]
            # if d > 0.5
            if len(self.f_index_gt) != 0:
                f[self.opposite_tensor[self.f_index_gt[:, 0]],
                self.f_index_gt[:, 1],
                self.f_index_gt[:, 2]] = (1 / (2 * self.d_gt)) * self.f_collided_gt[:, 0] + (
                        1 - 1 / (2 * self.d_gt)) * self.f_collided_gt[:, 1]

        if self.lattice.D == 3:
            # BOUNCE
            # if d <= 0.5
            if len(self.f_index_lt) != 0:
                f[self.opposite_tensor[self.f_index_lt[:, 0]],
                self.f_index_lt[:, 1],
                self.f_index_lt[:, 2],
                self.f_index_lt[:, 3]] = 2 * self.d_lt * self.f_collided_lt[:, 0] + (1 - 2 * self.d_lt) * f[
                    self.f_index_lt[:, 0],
                    self.f_index_lt[:, 1],
                    self.f_index_lt[:, 2],
                    self.f_index_lt[:, 3]]
            # if d > 0.5
            if len(self.f_index_gt) != 0:
                f[self.opposite_tensor[self.f_index_gt[:, 0]],
                self.f_index_gt[:, 1],
                self.f_index_gt[:, 2],
                self.f_index_gt[:, 3]] = (1 / (2 * self.d_gt)) * self.f_collided_gt[:, 0] + (
                        1 - 1 / (2 * self.d_gt)) * self.f_collided_gt[:, 1]

        # CALC. FORCE on boundary (MEM, MEA)
        if self.calc_force:
            self.calc_force_on_boundary(f)
        return f

    def make_no_stream_mask(self, f_shape):
        assert self.mask.shape == f_shape[1:]  # all dimensions of f except the 0th (q)
        # no_stream_mask has to be dimensions: (q,x,y,z) (z optional), but CAN be (x,y,z) (z optional).
        # ...in the latter case, torch.where broadcasts the mask to (q,x,y,z), so ALL q populations of a lattice-node are marked equally
        # return torch.tensor(self.mask, dtype=torch.bool)
        return self.lattice.convert_to_tensor(self.mask)

    def make_no_collision_mask(self, f_shape):
        # INFO: pay attention to the initialization of observable/moment-fields (u, rho,...) on the boundary nodes,
        # ...in the initial solution of your flow, especially if visualization or post processing uses the field-values
        # ...in the whole domain (including the boundary region)!
        assert self.mask.shape == f_shape[1:]
        # return torch.tensor(self.mask, dtype=torch.bool)  # self.lattice.convert_to_tensor(self.mask)
        return self.lattice.convert_to_tensor(self.mask)

    def calc_force_on_boundary(self, f_bounced):
        ### force = e * (f_collided + f_bounced[opp.])
        if self.lattice.D == 2:
            self.force_sum = torch.einsum('i..., id -> d',
                                          self.f_collided_lt[:, 0] + f_bounced[
                                              self.opposite_tensor[self.f_index_lt[:, 0]],
                                              self.f_index_lt[:, 1],
                                              self.f_index_lt[:, 2]],
                                          self.lattice.e[self.f_index_lt[:, 0]].float()) \
                             + torch.einsum('i..., id -> d',
                                            self.f_collided_gt[:, 0] + f_bounced[
                                                self.opposite_tensor[self.f_index_gt[:, 0]],
                                                self.f_index_gt[:, 1],
                                                self.f_index_gt[:, 2]],
                                            self.lattice.e[self.f_index_gt[:, 0]].float())
        if self.lattice.D == 3:
            self.force_sum = torch.einsum('i..., id -> d',
                                          self.f_collided_lt[:, 0] + f_bounced[
                                              self.opposite_tensor[self.f_index_lt[:, 0]],
                                              self.f_index_lt[:, 1],
                                              self.f_index_lt[:, 2],
                                              self.f_index_lt[:, 3]],
                                          self.lattice.e[self.f_index_lt[:, 0]].float()) \
                             + torch.einsum('i..., id -> d',
                                            self.f_collided_gt[:, 0] + f_bounced[
                                                self.opposite_tensor[self.f_index_gt[:, 0]],
                                                self.f_index_gt[:, 1],
                                                self.f_index_gt[:, 2],
                                                self.f_index_gt[:, 3]],
                                            self.lattice.e[self.f_index_gt[:, 0]].float())

    def store_f_collided(self, f_collided):
        for f_index_lgt, f_collided_lgt in zip([self.f_index_lt, self.f_index_gt],
                                               [self.f_collided_lt, self.f_collided_gt]):
            if len(f_index_lgt) != 0:
                for d in range(self.lattice.D):
                    indices = [f_index_lgt[:, 0],  # q
                               f_index_lgt[:, 1],  # x
                               f_index_lgt[:, 2]]  # y
                    if self.lattice.D == 3:
                        indices.append(f_index_lgt[:, 3])
                    f_collided_lgt[:, 0] = torch.clone(f_collided[indices])
                    indices[0] = self.opposite_tensor[f_index_lgt[:, 0]]
                    f_collided_lgt[:, 1] = torch.clone(f_collided[indices])
        # TODO: compare performance of THIS to original hardcoded "store_f_collided()" of IBB1, see below

    # >>> OLD version "semi hardcoded"
    # def store_f_collided(self, f_collided):
    #     if self.lattice.D == 2:
    #         if len(self.f_collided_lt) != 0:
    #             self.f_collided_lt[:, 0] = torch.clone(f_collided[self.f_index_lt[:, 0],  # q
    #                                                           self.f_index_lt[:, 1],  # x
    #                                                           self.f_index_lt[:, 2]])  # y
    #             self.f_collided_lt[:, 1] = torch.clone(f_collided[self.opposite_tensor[self.f_index_lt[:,0]],  # q
    #                                                           self.f_index_lt[:, 1],  # x
    #                                                           self.f_index_lt[:, 2]])  # y
    #         if len(self.f_collided_gt) != 0:
    #             self.f_collided_gt[:, 0] = torch.clone(f_collided[self.f_index_gt[:, 0],  # q
    #                                                           self.f_index_gt[:, 1],  # x
    #                                                           self.f_index_gt[:, 2]])  # y
    #             self.f_collided_gt[:, 1] = torch.clone(f_collided[self.opposite_tensor[self.f_index_gt[:,0]],  # q
    #                                                           self.f_index_gt[:, 1],  # x
    #                                                           self.f_index_gt[:, 2]])  # y
    #     if self.lattice.D == 3:
    #         if len(self.f_collided_lt) != 0:
    #             self.f_collided_lt[:, 0] = torch.clone(f_collided[self.f_index_lt[:, 0],  # q
    #                                                           self.f_index_lt[:, 1],  # x
    #                                                           self.f_index_lt[:, 2],  # y
    #                                                           self.f_index_lt[:, 3]])  # z
    #             self.f_collided_lt[:, 1] = torch.clone(f_collided[self.opposite_tensor[self.f_index_lt[:,0]],  # q
    #                                                           self.f_index_lt[:, 1],  # x
    #                                                           self.f_index_lt[:, 2],  # y
    #                                                           self.f_index_lt[:, 3]])  # z
    #         if len(self.f_collided_gt) != 0:
    #             self.f_collided_gt[:, 0] = torch.clone(f_collided[self.f_index_gt[:, 0],  # q
    #                                                               self.f_index_gt[:, 1],  # x
    #                                                               self.f_index_gt[:, 2],  # y
    #                                                               self.f_index_gt[:, 3]])  # z
    #             self.f_collided_gt[:, 1] = torch.clone(f_collided[self.opposite_tensor[self.f_index_gt[:, 0]],  # q
    #                                                               self.f_index_gt[:, 1],  # x
    #                                                               self.f_index_gt[:, 2],  # y
    #                                                               self.f_index_gt[:, 3]])  # z
    # <<< OLD version "semi hardcoded"

    def initialize_f_collided(self):
        f_collided_lt = torch.zeros_like(self.d_lt)  # float-tensor with number of (x_b nodes with d<=0.5) values
        f_collided_gt = torch.zeros_like(self.d_gt)  # float-tensor with number of (x_b nodes with d>0.5) values
        f_collided_lt_opposite = torch.zeros_like(self.d_lt)
        f_collided_gt_opposite = torch.zeros_like(self.d_gt)
        self.f_collided_lt = torch.stack((f_collided_lt, f_collided_lt_opposite), dim=1)
        self.f_collided_gt = torch.stack((f_collided_gt, f_collided_gt_opposite), dim=1)


class SlipBoundary:
    """bounces back in a direction given as 0, 1, or 2 for x, y, or z, respectively
        based on fullway bounce back algorithm (population remains in the wall for 1 time step)
    """

    def __init__(self, mask, lattice, direction):
        self.mask = lattice.convert_to_tensor(mask)
        self.lattice = lattice
        self.bb_direction = direction
        e = self.lattice.stencil.e
        bb_direction = self.bb_direction
        opposite_stencil = np.array(e)
        opposite_stencil[:, bb_direction] = -e[:, bb_direction]
        self.opposite = []
        for opp_dir in opposite_stencil:
            self.opposite.append(np.where(np.array(e == opp_dir).all(axis=1))[0][0])

    def __call__(self, f):
        f = torch.where(self.mask, f[self.opposite], f)
        return f

    def make_no_collision_mask(self, f_shape):
        assert self.mask.shape == f_shape[1:]
        return self.mask


class BounceBackBoundary:
    """Fullway Bounce-Back Boundary"""

    def __init__(self, mask, lattice):
        self.mask = lattice.convert_to_tensor(mask)
        self.lattice = lattice

    def __call__(self, f):
        f = torch.where(self.mask, f[self.lattice.stencil.opposite], f)
        return f

    def make_no_collision_mask(self, f_shape):
        assert self.mask.shape == f_shape[1:]
        return self.mask


class FullwayBounceBackBoundary:
    """Fullway Bounce-Back Boundary (with added force_on_boundary calculation)
    - fullway = inverts populations within two substeps
    - call() must be called after Streaming substep
    - calculates the force on the boundary:
        - calculation is done after streaming, but theoretically the force is evaluated based on the populations touching/crossing the boundary IN this streaming step
    """

    # based on Master-Branch "class BounceBackBoundary"
    # added option to calculate force on the boundary by Momentum Exchange Method

    def __init__(self, mask, lattice):
        self.mask = lattice.convert_to_tensor(mask)  # which nodes are solid
        self.lattice = lattice
        self.force_sum = torch.zeros_like(self.lattice.convert_to_tensor(self.lattice.stencil.e[0]))  # summed force vector on all boundary nodes, in D dimensions (x,y,(z))
        ### create f_mask, needed for force-calculation
        # ...(marks all fs which streamed into the boundary in prior streaming step)
        if self.lattice.D == 2:
            nx, ny = mask.shape  # domain size in x and y
            self.f_mask = np.zeros((self.lattice.Q, nx, ny), dtype=bool)
                # f_mask: [q, nx, ny], marks all fs on the boundary-border, which point into the boundary/solid
#            self.force = np.zeros((nx, ny, 2))  # force in x and y on all individual nodes
            a, b = np.where(mask)
                # np.arrays: list of (a) x-indizes and (b) y-indizes in the boundary.mask
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
                        if not mask[a[p] + self.lattice.stencil.e[i, 0] - border[0]*nx,
                                    b[p] + self.lattice.stencil.e[i, 1] - border[1]*ny]:
                            # if the neighbour of p is False in the boundary.mask, p is a solid node, neighbouring a fluid node:
                            # ...the direction pointing from the fluid neighbour to solid p is marked on the solid p
                            #OLD: self.f_mask[self.lattice.stencil.opposite[i], a[p] + self.lattice.stencil.e[i, 0], b[p] + self.lattice.stencil.e[i, 1]] = 1
                            self.f_mask[self.lattice.stencil.opposite[i], a[p], b[p]] = 1
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there
        if self.lattice.D == 3:  # like 2D, but in 3D...guess what...
            nx, ny, nz = mask.shape
            self.f_mask = np.zeros((self.lattice.Q, nx, ny, nz), dtype=bool)
#            self.force = np.zeros((nx, ny, nz, 3))
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
                            # OLD: self.f_mask[self.lattice.stencil.opposite[i], a[p] + self.lattice.stencil.e[i, 0], b[p] + self.lattice.stencil.e[i, 1], c[p] + self.lattice.stencil.e[i, 2]] = 1
                            self.f_mask[self.lattice.stencil.opposite[i], a[p], b[p], c[p]] = 1
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there
        self.f_mask = self.lattice.convert_to_tensor(self.f_mask)

    def __call__(self, f):
        # FULLWAY-BBBC: inverts populations on all boundary nodes

        # calc force on boundary:
        self.calc_force_on_boundary(f)
        # bounce (invert populations on boundary nodes)
        f = torch.where(self.mask, f[self.lattice.stencil.opposite], f)
        return f

    def make_no_collision_mask(self, f_shape):
        assert self.mask.shape == f_shape[1:]
        return self.mask

    def calc_force_on_boundary(self, f):
        # calculate force on boundary by momentum exchange method (MEA, MEM) according to Kruger et al., 2017, pp.215-217:
            # momentum (f_i*c_i - f_i_opposite*c_i_opposite = 2*f_i*c_i for a resting boundary) is summed for all...
            # ...populations pointing at the surface of the boundary
        tmp = torch.where(self.f_mask, f, torch.zeros_like(f))  # all populations f in the fluid region, which point at the boundary
        #self.force = 1 ** self.lattice.D * 2 * torch.einsum('i..., id -> d', tmp, self.lattice.e) / 1.0  # v1.1 - M.Kliemank
        #self.force = dx ** self.lattice.D * 2 * torch.einsum('i..., id -> d', tmp, self.lattice.e) / dx  # v.1.2 - M.Bille (dt=dx, dx as a parameter)
        self.force_sum = 2 * torch.einsum('i..., id -> d', tmp, self.lattice.e)  # CALCULATE FORCE / v2.0 - M.Bille: dx_lu = dt_lu is allways 1 (!)
            # explanation for 2D:
                # sums forces in x and in y (and z) direction,
                # tmp: all f, that are marked in f_mask
                    # tmp.size: 9 x nx x ny (for 2D)
                # self.lattice.e: 9 x 2 (for 2D)
                # - the multiplication of f_i and c_i is down through the first dimension (q) = direction, indexname i
                # - the sign is given by the coordinates of the stencil-vectors (e[0 to 8] for 2D)
                # -> results in two dimensional output (index d) for x- and y-direction (for 2D)
                # "dx**self-lattice.D" = dx³ (3D) or dx² (2D) as prefactor, converting momentum density to momentum
                    # theoretically DELTA P (difference in momentum density) is calculated
                    # assuming smooth momentum transfer over dt, force can be calculated through: F= dP/dt
                    # ...that's why theoretically dividing by dt=dx=1 is necessary (BUT: c_i=1=dx/dt=1 so that can be omitted (v2.0) !)

        # # calculate Force on all boundary nodes individually:
        # if self.lattice.D == 2:
        #     self.force = 2 * torch.einsum('qxy, qd -> xyd', tmp, self.lattice.e)  # force = [x-coordinate, y-coodrinate, direction (0=x, 1=y)]
        # if self.lattice.D == 3:
        #     self.force = 2 * torch.einsum('qxyz, qd -> xyzd', tmp, self.lattice.e)  # force = [x-coordinate, y-coodrinate, z-coodrinate, direction (0=x, 1=y, 2=z)]


class FullwayBounceBackBoundary_compact:

    def __init__(self, mask, lattice):
        self.mask = mask  # which nodes are solid
        self.lattice = lattice
        self.force_sum = torch.zeros_like(self.lattice.convert_to_tensor(self.lattice.stencil.e[0]))  # summed force vector on all boundary nodes, in D dimensions (x,y,(z))
        ### create f_mask, needed for force-calculation
        # ...(marks all fs which streamed into the boundary in prior streaming step)

        self.f_index = []

        if self.lattice.D == 2:
            nx, ny = mask.shape  # domain size in x and y

                # f_mask: [q, nx, ny], marks all fs on the boundary-border, which point into the boundary/solid
            a, b = np.where(mask)
                # np.arrays: list of (a) x-indizes and (b) y-indizes in the boundary.mask
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
                        if not mask[a[p] + self.lattice.stencil.e[i, 0] - border[0]*nx,
                                    b[p] + self.lattice.stencil.e[i, 1] - border[1]*ny]:
                            # if the neighbour of p is False in the boundary.mask, p is a solid node, neighbouring a fluid node:
                            # ...the direction pointing from the fluid neighbour to solid p is marked on the solid p

                            self.f_index.append([self.lattice.stencil.opposite[i], a[p], b[p]])  # list of [q, nx, ny], marks all fs on the boundary-border, which point into the boundary/solid
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
                            self.f_index.append([self.lattice.stencil.opposite[i], a[p], b[p], c[p]])
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there

        self.f_index = torch.tensor(np.array(self.f_index), device=self.lattice.device, dtype=torch.int64) # the batch-index has to be integer
        #PHILIPP_occ_angepasst? self.f_index = torch.tensor(self.f_index, device=self.lattice.device, dtype=torch.int64)  # the batch-index has to be integer
        self.opposite_tensor = torch.tensor(self.lattice.stencil.opposite, device=self.lattice.device,
                                            dtype=torch.int64)  # batch-index has to be a tensor

    def __call__(self, f):
        # FULLWAY-BBBC: inverts populations on all boundary nodes

        # calc force on boundary:
        self.calc_force_on_boundary(f)
        # bounce (invert populations on boundary nodes)
        # f = torch.where(self.mask, f[self.lattice.stencil.opposite], f)

        if self.lattice.D == 2:
            f[self.opposite_tensor[self.f_index[:, 0]],
              self.f_index[:, 1],
              self.f_index[:, 2]] = f[self.f_index[:, 0],
                                      self.f_index[:, 1],
                                      self.f_index[:, 2]]
        if self.lattice.D == 3:
            f[self.opposite_tensor[self.f_index[:, 0]],
              self.f_index[:, 1],
              self.f_index[:, 2],
              self.f_index[:, 3]] = f[self.f_index[:, 0],
                                      self.f_index[:, 1],
                                      self.f_index[:, 2],
                                      self.f_index[:, 3]]
        return f

    def make_no_collision_mask(self, f_shape):
        assert self.mask.shape == f_shape[1:]
        return self.lattice.convert_to_tensor(self.mask)

    def calc_force_on_boundary(self, f):
        if self.lattice.D == 2:
            self.force_sum = 2 * torch.einsum('i..., id -> d', f[self.f_index[:, 0],
                                                                 self.f_index[:, 1],
                                                                 self.f_index[:, 2]], self.lattice.e[self.f_index[:, 0]])
        if self.lattice.D == 3:
            self.force_sum = 2 * torch.einsum('i..., id -> d', f[self.f_index[:, 0],
                                                                 self.f_index[:, 1],
                                                                 self.f_index[:, 2],
                                                                 self.f_index[:, 3]], self.lattice.e[self.f_index[:, 0]])


class FullwayBounceBackBoundary_occ:

    def __init__(self, mask, lattice, solid_boundary_data: SolidBoundaryData = None, global_solid_mask=None, periodicity: tuple[bool,...] = None, calc_force=None):
        if not hasattr(solid_boundary_data, "solid_mask"):
            self.mask = mask
            self.solid_mask = mask
        else:
            self.mask = solid_boundary_data.solid_mask
            self.solid_mask = solid_boundary_data.solid_mask

        if periodicity is None:
            periodicity = (False, False, False if lattice.D == 3 else None)

        # TODO: correct periodicity and solid-to-solid contact: periodicity attribute and global solid mask! in neighbor search
        # global_solid_mask to filter out all "fake" fluid neighbors, which are outside the FWBB but not in the fluid region
        if global_solid_mask is None:
            global_solid_mask = np.zeros_like(self.mask, dtype=bool)
        other_solid_bc_mask = np.where(~self.mask, global_solid_mask, False)  # exclude self.mask from global_solid_mask

        self.lattice = lattice
        if calc_force is not None:
            self.force_sum = torch.zeros_like(self.lattice.convert_to_tensor(
                self.lattice.stencil.e[0]))  # summed force vector on all boundary nodes, in D dimensions (x,y,(z))
            self.calc_force = True
        else:
            self.calc_force = False

        ### create f_mask_fwbb, needed for force-calculation
        # ...(marks all fs which streamed into the boundary in prior streaming step)
        # ... in other words: marks all fs that need to be bounced
        self.f_index_fwbb = []

        if self.lattice.D == 2:
            nx, ny = mask.shape  # domain size in x and y

                # f_mask: [q, nx, ny], marks all fs on the boundary-border, which point into the boundary/solid
            ix_sp, iy_sp = np.where(mask)
                # np.arrays: list of (ix_sp) x-indizes and (iy_sp) y-indizes in the boundary.mask
                # ...to enable iteration over all boundary/wall/object-nodes
            for sp_index in range(0, len(ix_sp)):  # for all TRUE-nodes in boundary.mask
                for q_index in range(0, self.lattice.Q):  # for all stencil-directions c_i (lattice.stencil.e in lettuce)
                    # check for boundary-nodes neighboring the domain-border.
                    # ...they have to take the periodicity into account...
                    border = np.zeros(self.lattice.D, dtype=int)
                    if ix_sp[sp_index] == 0 and self.lattice.stencil.e[q_index, 0] == -1 and periodicity[0]:  # searching border on left
                        border[0] = -1
                    elif ix_sp[sp_index] == nx - 1 and self.lattice.e[q_index, 0] == 1 and periodicity[0]:  # searching border on right
                        border[0] = 1
                    if iy_sp[sp_index] == 0 and self.lattice.stencil.e[q_index, 1] == -1 and periodicity[1]:  # searching border on left
                        border[1] = -1
                    elif iy_sp[sp_index] == ny - 1 and self.lattice.e[q_index, 1] == 1 and periodicity[1]:  # searching border on right
                        border[1] = 1
                    try:  # try in case the neighboring cell does not exist (= an f pointing out of the simulation domain)
                        if (not mask[ix_sp[sp_index] + self.lattice.stencil.e[q_index, 0] - border[0]*nx,
                                    iy_sp[sp_index] + self.lattice.stencil.e[q_index, 1] - border[1]*ny]
                                and not other_solid_bc_mask[ix_sp[sp_index] + self.lattice.stencil.e[q_index, 0] - border[0]*nx,
                                    iy_sp[sp_index] + self.lattice.stencil.e[q_index, 1] - border[1]*ny]):
                            # if the neighbour of sp_index is False in the boundary.mask, sp_index is ix_sp solid node, neighbouring ix_sp fluid node:
                            # ...the direction pointing from the fluid neighbour to solid sp_index is marked on the solid sp_index

                            self.f_index_fwbb.append([self.lattice.stencil.opposite[q_index], ix_sp[sp_index], iy_sp[sp_index]])  # list of [q, nx, ny], marks all fs on the boundary-border, which point into the boundary/solid
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there
        if self.lattice.D == 3:  # like 2D, but in 3D...guess what...
            nx, ny, nz = mask.shape

            ix_sp, iy_sp, c = np.where(mask)
            for sp_index in range(0, len(ix_sp)):
                for q_index in range(0, self.lattice.Q):
                    border = np.zeros(self.lattice.D, dtype=int)
                    if ix_sp[sp_index] == 0 and self.lattice.stencil.e[q_index, 0] == -1 and periodicity[0]:  # searching border on left
                        border[0] = -1
                    elif ix_sp[sp_index] == nx - 1 and self.lattice.e[q_index, 0] == 1 and periodicity[0]:  # searching border on right
                        border[0] = 1
                    if iy_sp[sp_index] == 0 and self.lattice.stencil.e[q_index, 1] == -1 and periodicity[1]:  # searching border on left
                        border[1] = -1
                    elif iy_sp[sp_index] == ny - 1 and self.lattice.e[q_index, 1] == 1 and periodicity[1]:  # searching border on right
                        border[1] = 1
                    if c[sp_index] == 0 and self.lattice.stencil.e[q_index, 2] == -1 and periodicity[2]:  # searching border on left
                        border[2] = -1
                    elif c[sp_index] == nz - 1 and self.lattice.e[q_index, 2] == 1 and periodicity[2]:  # searching border on right
                        border[2] = 1
                    try:  # try in case the neighboring cell does not exist (and f pointing out of simulation domain)
                        if (not mask[ix_sp[sp_index] + self.lattice.stencil.e[q_index, 0] - border[0] * nx,
                                    iy_sp[sp_index] + self.lattice.stencil.e[q_index, 1] - border[1] * ny,
                                    c[sp_index] + self.lattice.stencil.e[q_index, 2] - border[2] * nz]
                                and not other_solid_bc_mask[ix_sp[sp_index] + self.lattice.stencil.e[q_index, 0] - border[0] * nx,
                                    iy_sp[sp_index] + self.lattice.stencil.e[q_index, 1] - border[1] * ny,
                                    c[sp_index] + self.lattice.stencil.e[q_index, 2] - border[2] * nz]):
                            self.f_index_fwbb.append([self.lattice.stencil.opposite[q_index], ix_sp[sp_index], iy_sp[sp_index], c[sp_index]])
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there

        self.f_index_fwbb = torch.tensor(np.array(self.f_index_fwbb), device=self.lattice.device, dtype=torch.int64) # the batch-index has to be integer
        #PHILIPP_occ_angepasst? self.f_index = torch.tensor(self.f_index, device=self.lattice.device, dtype=torch.int64)  # the batch-index has to be integer
        self.opposite_tensor = torch.tensor(self.lattice.stencil.opposite, device=self.lattice.device,
                                            dtype=torch.int64)  # batch-index has to be ix_sp tensor


    def __call__(self, f):
        # FULLWAY-BBBC: inverts populations on all boundary nodes

        # calc force on boundary:#
        if self.calc_force:
            self.calc_force_on_boundary(f)
        # bounce (invert populations on boundary nodes)
        # f = torch.where(self.mask, f[self.lattice.stencil.opposite], f)

        if self.lattice.D == 2:
            f[self.opposite_tensor[self.f_index_fwbb[:, 0]],
            self.f_index_fwbb[:, 1],
            self.f_index_fwbb[:, 2]] = f[self.f_index_fwbb[:, 0],
            self.f_index_fwbb[:, 1],
            self.f_index_fwbb[:, 2]]
        if self.lattice.D == 3:
            f[self.opposite_tensor[self.f_index_fwbb[:, 0]],
            self.f_index_fwbb[:, 1],
            self.f_index_fwbb[:, 2],
            self.f_index_fwbb[:, 3]] = f[self.f_index_fwbb[:, 0],
            self.f_index_fwbb[:, 1],
            self.f_index_fwbb[:, 2],
            self.f_index_fwbb[:, 3]]
        return f

    def make_no_collision_mask(self, f_shape):
        assert self.mask.shape == f_shape[1:]
        return self.lattice.convert_to_tensor(self.mask)

    def calc_force_on_boundary(self, f):
        if self.lattice.D == 2:
            self.force_sum = 2 * torch.einsum('i..., id -> d', f[self.f_index_fwbb[:, 0],
            self.f_index_fwbb[:, 1],
            self.f_index_fwbb[:, 2]], self.lattice.e[self.f_index_fwbb[:, 0]])
        if self.lattice.D == 3:
            self.force_sum = 2 * torch.einsum('i..., id -> d', f[self.f_index_fwbb[:, 0],
            self.f_index_fwbb[:, 1],
            self.f_index_fwbb[:, 2],
            self.f_index_fwbb[:, 3]], self.lattice.e[self.f_index_fwbb[:, 0]])

class HalfwayBounceBackBoundary:
    """Halfway Bounce Back Boundary (with added force_on_boundary calculation)
    - halfway = inverts populations within one substep
    - call() must be called after Streaming substep
    - calculates the force on the boundary:
        - calculation is done after streaming, but theoretically the force is evaluated based on the populations touching/crossing the boundary IN this streaming step
    """

    def __init__(self, mask, lattice):
        self.mask = mask
        self.lattice = lattice
        self.force_sum = torch.zeros_like(self.lattice.convert_to_tensor(self.lattice.stencil.e[0]))  # summed force vector on all boundary nodes, in D dimensions (x,y,(z))
        ### create f_mask, needed for force-calculation
        # ...(marks all fs which point from fluid to solid (boundary))
        if self.lattice.D == 2:
            nx, ny = mask.shape  # domain size in x and y
            self.f_mask = np.zeros((self.lattice.Q, nx, ny), dtype=bool)
                # f_mask: [q, nx, ny], marks all fs which point from fluid to solid (boundary)
#            self.force = np.zeros((nx, ny, 2))  # force in x and y on all individual nodes
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
                        if not mask[a[p] + self.lattice.stencil.e[i, 0] - border[0]*nx,
                                    b[p] + self.lattice.stencil.e[i, 1] - border[1]*ny]:
                            # if the neighbour of p is False in the boundary.mask, p is a solid node, neighbouring a fluid node:
                            # ...the direction pointing from the fluid neighbour to solid p is marked on the neighbour
                            self.f_mask[self.lattice.stencil.opposite[i],
                                        a[p] + self.lattice.stencil.e[i, 0] - border[0]*nx,
                                        b[p] + self.lattice.stencil.e[i, 1] - border[1]*ny] = 1
                            # f_mask[q,x,y]
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there
        if self.lattice.D == 3:  # like 2D, but in 3D...guess what...
            nx, ny, nz = mask.shape
            self.f_mask = np.zeros((self.lattice.Q, nx, ny, nz), dtype=bool)
#            self.force = np.zeros((nx, ny, nz, 3))
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
                            self.f_mask[self.lattice.stencil.opposite[i],
                                        a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                        b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                        c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz] = 1
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there
        self.f_mask = self.lattice.convert_to_tensor(self.f_mask)
        self.f_collided = torch.zeros_like(self.f_mask, device=self.lattice.device, dtype=self.lattice.dtype)

    def __call__(self, f):
        # HALFWAY-BB: overwrite all populations (on fluid nodes) which came from boundary with pre-streaming populations (on fluid nodes) which pointed at boundary
            #print("f_mask:\n", self.f_mask)
            #print("f_mask(q2,x1,y1):\n", self.f_mask[2, 1, 1])
            #print("f_mask(q2,x1,y3):\n", self.f_mask[2, 1, 3])
            #print("f_mask(opposite):\n", self.f_mask[self.lattice.stencil.opposite])
        # calc force on boundary:
        self.calc_force_on_boundary()
        # bounce (invert populations on fluid nodes neighboring solid nodes)
        f = torch.where(self.f_mask[self.lattice.stencil.opposite], self.f_collided[self.lattice.stencil.opposite], f)
        # ersetze alle "von der boundary kommenden" Populationen durch ihre post-collision_pre-streaming entgegengesetzten Populationen
        # ...bounce-t die post_collision/pre-streaming Populationen an der Boundary innerhalb eines Zeitschrittes
        # ...von außen betrachtet wird "während des streamings", innerhalb des gleichen Zeitschritts invertiert.
        # (?) es wird keine no_stream_mask benötigt, da sowieso alles, was aus der boundary geströmt käme hier durch pre-Streaming Populationen überschrieben wird.
        # ...ist das so, oder entsteht dadurch "Strömung" innerhalb des Obstacles? Diese hat zwar keinen direkten Einfluss auf die Größen im Fluidbereich,
        # ... lässt aber in der Visualisierung Werte ungleich Null innerhalb von Objekten entstehen und Mittelwerte etc. könnten davon beeinflusst werden. (?)
        return f

    def make_no_stream_mask(self, f_shape):
        assert self.mask.shape == f_shape[1:]  # all dimensions of f except the 0th (q)
            # no_stream_mask has to be dimensions: (q,x,y,z) (z optional), but CAN be (x,y,z) (z optional).
            # ...in the latter case, torch.where broadcasts the mask to (q,x,y,z), so ALL q populations of a lattice-node are marked equally
        return self.lattice.convert_to_tensor(self.mask)

    def make_no_collision_mask(self, f_shape):
        # INFO: for the halfway bounce back boundary, a no_collision_mask ist not necessary, because the no_stream_mask
        # ...prevents interaction between nodes inside and outside the boundary region.
        # INFO: pay attention to the initialization of observable/moment-fields (u, rho,...) on the boundary nodes,
        # ...in the initial solution of your flow, especially if visualization or post-processing uses the field-values
        # ...in the whole domain (including the boundary region)!
        assert self.mask.shape == f_shape[1:]
        return self.lattice.convert_to_tensor(self.mask)

    def calc_force_on_boundary(self):
        # calculate force on boundary by momentum exchange method (MEA, MEM) according to Kruger et al., 2017, pp.215-217:
            # momentum (f_i*c_i - f_i_opposite*c_i_opposite = 2*f_i*c_i for a resting boundary) is summed for all...
            # ...populations pointing at the surface of the boundary
        tmp = torch.where(self.f_mask, self.f_collided, torch.zeros_like(self.f_collided))  # all populations f in the fluid region, which point at the boundary
        #self.force = 1 ** self.lattice.D * 2 * torch.einsum('i..., id -> d', tmp, self.lattice.e) / 1.0  # v1.1 - M.Kliemank
        #self.force = dx ** self.lattice.D * 2 * torch.einsum('i..., id -> d', tmp, self.lattice.e) / dx  # v.1.2 - M.Bille (dt=dx, dx as a parameter)
        self.force_sum = 2 * torch.einsum('i..., id -> d', tmp, self.lattice.e)  # CALCULATE FORCE / v2.0 - M.Bille: dx_lu = dt_lu is allways 1 (!)
            # explanation for 2D:
                # sums forces in x and in y (and z) direction,
                # tmp: all f, that are marked in f_mask
                    # tmp.size: 9 x nx x ny (for 2D)
                # self.lattice.e: 9 x 2 (for 2D)
                # - the multiplication of f_i and c_i is down through the first dimension (q) = direction, indexname i
                # - the sign is given by the coordinates of the stencil-vectors (e[0 to 8] for 2D)
                # -> results in two dimensional output (index d) for x- and y-direction (for 2D)
                # "dx**self-lattice.D" = dx³ (3D) or dx² (2D) as prefactor, converting momentum density to momentum
                    # theoretically DELTA P (difference in momentum density) is calculated
                    # assuming smooth momentum transfer over dt, force can be calculated through: F= dP/dt
                    # ...that's why theoretically dividing by dt=dx=1 is necessary (BUT: c_i=1=dx/dt=1 so that can be omitted (v2.0) !)

        # # calculate Force on all boundary nodes individually:
        # if self.lattice.D == 2:
        #     self.force = 2 * torch.einsum('qxy, qd -> xyd', tmp,
        #                                   self.lattice.e)  # force = [x-coordinate, y-coodrinate, direction (0=x, 1=y)]
        # if self.lattice.D == 3:
        #     self.force = 2 * torch.einsum('qxyz, qd -> xyzd', tmp, self.lattice.e)  # force = [x-coordinate, y-coodrinate, z-coodrinate, direction (0=x, 1=y, 2=z)]

    def store_f_collided(self, f_collided):
        self.f_collided = torch.clone(f_collided)

class HalfwayBounceBackBoundary_compact_v1:

    def __init__(self, mask, lattice):
        self.mask = mask
        self.lattice = lattice
        self.force_sum = torch.zeros_like(self.lattice.convert_to_tensor(self.lattice.stencil.e[0]))  # summed force vector on all boundary nodes, in D dimensions (x,y,(z))
        ### create f_mask, needed for force-calculation
        # ...(marks all fs which point from fluid to solid (boundary))

        self.f_index = []

        if self.lattice.D == 2:
            nx, ny = mask.shape  # domain size in x and y
            self.f_mask = np.zeros((self.lattice.Q, nx, ny),
                                   dtype=bool)  # f_mask: [q, nx, ny], marks all fs which point from fluid to solid (boundary), needed to collect f_collided in simulation
                # f_mask: [q, nx, ny], marks all fs which point from fluid to solid (boundary)

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
                        if not mask[a[p] + self.lattice.stencil.e[i, 0] - border[0]*nx,
                                    b[p] + self.lattice.stencil.e[i, 1] - border[1]*ny]:
                            # if the neighbour of p is False in the boundary.mask, p is a solid node, neighbouring a fluid node:
                            # ...the direction pointing from the fluid neighbour to solid p is marked on the neighbour
                            self.f_mask[self.lattice.stencil.opposite[i],
                                        a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                        b[p] + self.lattice.stencil.e[i, 1] - border[
                                            1] * ny] = 1  # f_mask[q,x,y], marks all fs which point from fluid to solid (boundary)

                            self.f_index.append([self.lattice.stencil.opposite[i],
                                                a[p] + self.lattice.stencil.e[i, 0] - border[0]*nx,
                                                b[p] + self.lattice.stencil.e[i, 1] - border[1]*ny])

                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there
        if self.lattice.D == 3:  # like 2D, but in 3D...guess what...
            nx, ny, nz = mask.shape
            self.f_mask = np.zeros((self.lattice.Q, nx, ny, nz), dtype=bool)
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
                            self.f_mask[self.lattice.stencil.opposite[i],
                                        a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                        b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                        c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz] = 1
                            self.f_index.append([self.lattice.stencil.opposite[i],
                                                a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                                c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz])
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there
        self.f_mask = self.lattice.convert_to_tensor(self.f_mask)
        self.f_index = torch.tensor(np.array(self.f_index), device=self.lattice.device, dtype=torch.int64)  # the batch-index has to be integer
        self.opposite_tensor = torch.tensor(self.lattice.stencil.opposite, device=self.lattice.device,
                                            dtype=torch.int64)  # batch-index has to be a tensor

        self.f_collided = []

        if lattice.D == 2:
            fc_q, fc_x, fc_y = torch.where(self.f_mask + self.f_mask[self.lattice.stencil.opposite])  # which populations of f_collided to store
            self.fc_index = torch.stack((fc_q, fc_x, fc_y))
        if lattice.D == 3:
            fc_q, fc_x, fc_y, fc_z = torch.where(self.f_mask + self.f_mask[self.lattice.stencil.opposite])  # which populations of f_collided to store
            self.fc_index = torch.stack((fc_q, fc_x, fc_y, fc_z))

    def __call__(self, f):
        # calc force on boundary:
        self.calc_force_on_boundary()
        # bounce (invert populations on fluid nodes neighboring solid nodes)
        # f = torch.where(self.f_mask[self.lattice.stencil.opposite], f_collided[self.lattice.stencil.opposite], f)

        if self.lattice.D == 2:
            f[self.opposite_tensor[self.f_index[:, 0]],
              self.f_index[:, 1],
              self.f_index[:, 2]] = self.f_collided.to_dense()[self.f_index[:, 0],
                                                          self.f_index[:, 1],
                                                          self.f_index[:, 2]]
        if self.lattice.D == 3:
            f[self.opposite_tensor[self.f_index[:, 0]],
              self.f_index[:, 1],
              self.f_index[:, 2],
              self.f_index[:, 3]] = self.f_collided.to_dense()[self.f_index[:, 0],
                                                          self.f_index[:, 1],
                                                          self.f_index[:, 2],
                                                          self.f_index[:, 3]]
        return f

    def make_no_stream_mask(self, f_shape):
        assert self.mask.shape == f_shape[1:]  # all dimensions of f except the 0th (q)
            # no_stream_mask has to be dimensions: (q,x,y,z) (z optional), but CAN be (x,y,z) (z optional).
            # ...in the latter case, torch.where broadcasts the mask to (q,x,y,z), so ALL q populations of a lattice-node are marked equally
        return self.lattice.convert_to_tensor(self.mask)

    def make_no_collision_mask(self, f_shape):
        # INFO: for the halfway bounce back boundary, a no_collision_mask ist not necessary, because the no_streaming_mask
        # ...prevents interaction between nodes inside and outside the boundary region.
        # INFO: pay attention to the initialization of observable/moment-fields (u, rho,...) on the boundary nodes,
        # ...in the initial solution of your flow, especially if visualization or post-processing uses the field-values
        # ...in the whole domain (including the boundary region)!
        assert self.mask.shape == f_shape[1:]
        return self.lattice.convert_to_tensor(self.mask)

    def calc_force_on_boundary(self):
        # calculate force on boundary by momentum exchange method (MEA, MEM) according to Kruger et al., 2017, pp.215-217:
            # momentum (f_i*c_i - f_i_opposite*c_i_opposite = 2*f_i*c_i for a resting boundary) is summed for all...
            # ...populations pointing at the surface of the boundary
        if self.lattice.D == 2:
            self.force_sum = 2 * torch.einsum('i..., id -> d', self.f_collided.to_dense()[self.f_index[:, 0],
                                                                                     self.f_index[:, 1],
                                                                                     self.f_index[:, 2]],
                                              self.lattice.e[self.f_index[:, 0]])
        if self.lattice.D == 3:
            self.force_sum = 2 * torch.einsum('i..., id -> d', self.f_collided.to_dense()[self.f_index[:, 0],
                                                                                     self.f_index[:, 1],
                                                                                     self.f_index[:, 2],
                                                                                     self.f_index[:, 3]],
                                              self.lattice.e[self.f_index[:, 0]])

    def store_f_collided(self, f_collided):
        if self.lattice.D == 2:
            self.f_collided = torch.clone(torch.sparse_coo_tensor(indices=self.fc_index,
                                                                  values=f_collided[self.fc_index[0], self.fc_index[1],
                                                                                    self.fc_index[2]],
                                                                  size=f_collided.size()))
        if self.lattice.D == 3:
            self.f_collided = torch.clone(torch.sparse_coo_tensor(indices=self.fc_index,
                                                                  values=f_collided[self.fc_index[0], self.fc_index[1],
                                                                                    self.fc_index[2], self.fc_index[3]],
                                                                  size=f_collided.size()))

class HalfwayBounceBackBoundary_compact_v2:

    def __init__(self, mask, lattice, calc_force = False):
        self.mask = mask  # location of solid-nodes
        self.lattice = lattice
        self.force_sum = torch.zeros_like(self.lattice.convert_to_tensor(
            self.lattice.stencil.e[0]))  # summed force vector on all boundary nodes, in D dimensions (x,y,(z))

        self.f_index = []

        # searching boundary-fluid-interface and append indices to f_index, distance to boundary to d
        if self.lattice.D == 2:
            nx, ny = mask.shape  # domain size in x and y
            a, b = np.where(mask)  # x- and y-index of boundaryTRUE nodes for iteration over boundary area

            for p in range(0, len(a)):  # for all TRUE-nodes in boundary.mask
                for i in range(0, self.lattice.Q):  # for all stencil-directions c_i (lattice.stencil.e in lettuce)
                    # check for boundary-nodes neighboring the domain-border.
                    # ...they have to take the periodicity into account...
                    border = np.zeros(self.lattice.D, dtype=int)

                    if a[p] == 0 and self.lattice.stencil.e[i, 0] == -1:  # searching border on left [x]
                        border[0] = -1
                    elif a[p] == nx - 1 and self.lattice.e[i, 0] == 1:  # searching border on right [x]
                        border[0] = 1

                    if b[p] == 0 and self.lattice.stencil.e[i, 1] == -1:  # searching border on left [y]
                        border[1] = -1
                    elif b[p] == ny - 1 and self.lattice.e[i, 1] == 1:  # searching border on right [y]
                        border[1] = 1

                    try:  # try in case the neighboring cell does not exist (= an f pointing out of the simulation domain)
                        if not mask[a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                    b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny]:
                            # if the neighbour of p is False in the boundary.mask, p is a solid node, neighbouring a fluid node:
                            # ...the direction pointing from the fluid neighbour to solid p is marked on the neighbour

                            self.f_index.append([self.lattice.stencil.opposite[i],
                                                 a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                 b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny])
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there

        if self.lattice.D == 3:  # like 2D, but in 3D...guess what...
            nx, ny, nz = mask.shape
            a, b, c = np.where(mask)

            for p in range(0, len(a)):
                for i in range(0, self.lattice.Q):
                    border = np.zeros(self.lattice.D, dtype=int)
                    # x - direction
                    if a[p] == 0 and self.lattice.stencil.e[i, 0] == -1:  # searching border on left
                        border[0] = -1
                    elif a[p] == nx - 1 and self.lattice.e[i, 0] == 1:  # searching border on right
                        border[0] = 1
                    # y - direction
                    if b[p] == 0 and self.lattice.stencil.e[i, 1] == -1:  # searching border on left
                        border[1] = -1
                    elif b[p] == ny - 1 and self.lattice.e[i, 1] == 1:  # searching border on right
                        border[1] = 1
                    # z - direction
                    if c[p] == 0 and self.lattice.stencil.e[i, 2] == -1:  # searching border on left
                        border[2] = -1
                    elif c[p] == nz - 1 and self.lattice.e[i, 2] == 1:  # searching border on right
                        border[2] = 1

                    try:  # try in case the neighboring cell does not exist (an f pointing out of simulation domain)
                        if not mask[a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                    b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                    c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz]:


                                self.f_index.append([self.lattice.stencil.opposite[i],
                                                     a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                     b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                                     c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz])
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there

        # convert relevant tensors:
        self.f_index = torch.tensor(np.array(self.f_index), device=self.lattice.device,
                                       dtype=torch.int64)  # the batch-index has to be integer
        self.opposite_tensor = torch.tensor(self.lattice.stencil.opposite, device=self.lattice.device,
                                            dtype=torch.int64)  # batch-index has to be a tensor
        # f_collided = torch.zeros_like(self.f_index[:, 0], dtype=self.lattice.dtype)
        # f_collided_opposite = torch.zeros_like(self.f_index[:, 0], dtype=self.lattice.dtype)
        # self.f_collided = torch.stack((f_collided, f_collided_opposite), dim=1)

    def __call__(self, f):
        # calc force on boundary:
        self.calc_force_on_boundary()
        # bounce (invert populations on fluid nodes neighboring solid nodes)
        # f = torch.where(self.f_mask[self.lattice.stencil.opposite], f_collided[self.lattice.stencil.opposite], f)

        if self.lattice.D == 2:
            f[self.opposite_tensor[self.f_index[:, 0]],
              self.f_index[:, 1],
              self.f_index[:, 2]] = self.f_collided[:, 0]
        if self.lattice.D == 3:
            f[self.opposite_tensor[self.f_index[:, 0]],
              self.f_index[:, 1],
              self.f_index[:, 2],
              self.f_index[:, 3]] = self.f_collided[:, 0]
        return f

    def make_no_stream_mask(self, f_shape):
        assert self.mask.shape == f_shape[1:]  # all dimensions of f except the 0th (q)
        # no_stream_mask has to be dimensions: (q,x,y,z) (z optional), but CAN be (x,y,z) (z optional).
        # ...in the latter case, torch.where broadcasts the mask to (q,x,y,z), so ALL q populations of a lattice-node are marked equally
        return self.lattice.convert_to_tensor(self.mask)

    def make_no_collision_mask(self, f_shape):
        # INFO: for the halfway bounce back boundary, a no_collision_mask ist not necessary, because the no_stream_mask
        # ...prevents interaction between nodes inside and outside of the boundary region.
        # INFO: pay attention to the initialization of observable/moment-fields (u, rho,...) on the boundary nodes,
        # ...in the initial solution of your flow, especially if visualization or post processing uses the field-values
        # ...in the whole domain (including the boundary region)!
        assert self.mask.shape == f_shape[1:]
        return self.lattice.convert_to_tensor(self.mask)

    def calc_force_on_boundary(self):
        # calculate force on boundary by momentum exchange method (MEA, MEM) according to Kruger et al., 2017, pp.215-217:
        # momentum (f_i*c_i - f_i_opposite*c_i_opposite = 2*f_i*c_i for a resting boundary) is summed for all...
        # ...populations pointing at the surface of the boundary
        self.force_sum = 2 * torch.einsum('i..., id -> d', self.f_collided[:, 0], self.lattice.e[self.f_index[:, 0]])

    def store_f_collided(self, f_collided):
        if self.lattice.D == 2:
            self.f_collided[:, 0] = torch.clone(f_collided[self.f_index[:, 0],  # q
                                                          self.f_index[:, 1],  # x
                                                          self.f_index[:, 2]])  # y
            self.f_collided[:, 1] = torch.clone(f_collided[self.opposite_tensor[self.f_index[:,0]],  # q
                                                          self.f_index[:, 1],  # x
                                                          self.f_index[:, 2]])  # y
        if self.lattice.D == 3:
            self.f_collided[:, 0] = torch.clone(f_collided[self.f_index[:, 0],  # q
                                                          self.f_index[:, 1],  # x
                                                          self.f_index[:, 2],  # y
                                                          self.f_index[:, 3]])  # z
            self.f_collided[:, 1] = torch.clone(f_collided[self.opposite_tensor[self.f_index[:,0]],  # q
                                                          self.f_index[:, 1],  # x
                                                          self.f_index[:, 2],  # y
                                                          self.f_index[:, 3]])  # z

    def initialize_f_collided(self):
        f_collided = torch.zeros_like(self.f_index[:, 0], dtype=self.lattice.dtype)
        f_collided_opposite = torch.zeros_like(self.f_index[:, 0], dtype=self.lattice.dtype)
        self.f_collided = torch.stack((f_collided, f_collided_opposite), dim=1)

class HalfwayBounceBackBoundary_occ:

    def __init__(self, mask, lattice, solid_boundary_data: SolidBoundaryData = None, global_solid_mask=None, periodicity: tuple[bool,...] = None, calc_force=None):
        if not hasattr(solid_boundary_data, "solid_mask"):
            self.mask = mask
            self.solid_mask = mask
        else:
            self.mask = solid_boundary_data.solid_mask
            self.solid_mask = solid_boundary_data.solid_mask

        # global_solid_mask to filter out all "fake" fluid neighbors, which are outside this HWBB but not in the fluid region
        if global_solid_mask is None:
            global_solid_mask = self.mask

        if periodicity is None:
            periodicity = (False, False, False if lattice.D == 3 else None)

        self.lattice = lattice
        if calc_force is not None:
            self.force_sum = torch.zeros_like(self.lattice.convert_to_tensor(
                self.lattice.stencil.e[0]))  # summed force vector on all boundary nodes, in D dimensions (x,y,(z))
            self.calc_force = True
        else:
            self.calc_force = False

        self.f_index = []

        # combine f_index_lt and f_index_gt to self.f_index
        if (hasattr(solid_boundary_data, "f_index_gt") or hasattr(solid_boundary_data, "f_index_lt")) and len(solid_boundary_data.f_index_lt.shape) == len(solid_boundary_data.f_index_gt.shape):  # if solid_boundary_data contains batch_indices, use them
            self.f_index = np.concatenate((solid_boundary_data.f_index_lt, solid_boundary_data.f_index_gt), axis=0)
        elif hasattr(solid_boundary_data, "f_index_gt") and solid_boundary_data.f_index_lt.shape[0] == 0:
            self.f_index = solid_boundary_data.f_index_gt
        elif hasattr(solid_boundary_data, "f_index_lt") and solid_boundary_data.f_index_gt.shape[0] == 0:
            self.f_index = solid_boundary_data.f_index_lt
        else:  #else do ghetto-neighbour_search below
            print("(INFO) HWBB didn't find solid_boundary_data, doing legacy neighbour_search on mask...")
            # searching boundary-fluid-interface and append indices to f_index, distance to boundary to d
            if self.lattice.D == 2:
                nx, ny = mask.shape  # domain size in x and y
                a, b = np.where(mask)  # x- and y-index of boundaryTRUE nodes for iteration over boundary area

                for p in range(0, len(a)):  # for all TRUE-nodes in boundary.mask
                    for i in range(0, self.lattice.Q):  # for all stencil-directions c_i (lattice.stencil.e in lettuce)
                        # check for boundary-nodes neighboring the domain-border.
                        # ...they have to take the periodicity into account...
                        border = np.zeros(self.lattice.D, dtype=int)

                        if a[p] == 0 and self.lattice.stencil.e[i, 0] == -1 and periodicity[0]:  # searching border on left [x]
                            border[0] = -1
                        elif a[p] == nx - 1 and self.lattice.e[i, 0] == 1 and periodicity[0]:  # searching border on right [x]
                            border[0] = 1

                        if b[p] == 0 and self.lattice.stencil.e[i, 1] == -1 and periodicity[1]:  # searching border on left [y]
                            border[1] = -1
                        elif b[p] == ny - 1 and self.lattice.e[i, 1] == 1 and periodicity[1]:  # searching border on right [y]
                            border[1] = 1

                        try:  # try in case the neighboring cell does not exist (= an f pointing out of the simulation domain)
                            if (not mask[a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                        b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny]
                                and not global_solid_mask[
                                    a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                    b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny]):
                                # if the neighbour of p is False in the boundary.mask, p is a solid node, neighbouring a fluid node:
                                # ...the direction pointing from the fluid neighbour to solid p is marked on the neighbour

                                self.f_index.append([self.lattice.stencil.opposite[i],
                                                     a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                     b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny])
                        except IndexError:
                            pass  # just ignore this iteration since there is no neighbor there

            if self.lattice.D == 3:  # like 2D, but in 3D...guess what...
                nx, ny, nz = mask.shape
                a, b, c = np.where(mask)

                for p in range(0, len(a)):
                    for i in range(0, self.lattice.Q):
                        border = np.zeros(self.lattice.D, dtype=int)
                        # x - direction
                        if a[p] == 0 and self.lattice.stencil.e[i, 0] == -1 and periodicity[0]:  # searching border on left
                            border[0] = -1
                        elif a[p] == nx - 1 and self.lattice.e[i, 0] == 1 and periodicity[0]:  # searching border on right
                            border[0] = 1
                        # y - direction
                        if b[p] == 0 and self.lattice.stencil.e[i, 1] == -1 and periodicity[1]:  # searching border on left
                            border[1] = -1
                        elif b[p] == ny - 1 and self.lattice.e[i, 1] == 1 and periodicity[1]:  # searching border on right
                            border[1] = 1
                        # z - direction
                        if c[p] == 0 and self.lattice.stencil.e[i, 2] == -1 and periodicity[2]:  # searching border on left
                            border[2] = -1
                        elif c[p] == nz - 1 and self.lattice.e[i, 2] == 1 and periodicity[2]:  # searching border on right
                            border[2] = 1

                        try:  # try in case the neighboring cell does not exist (an f pointing out of simulation domain)
                            if (not mask[a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                        b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                        c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz]
                                and not global_solid_mask[
                                    a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                    b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                    c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz]):

                                    self.f_index.append([self.lattice.stencil.opposite[i],
                                                         a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                         b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                                         c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz])
                        except IndexError:
                            pass  # just ignore this iteration since there is no neighbor there

        # convert relevant tensors:
        self.f_index = torch.tensor(np.array(self.f_index), device=self.lattice.device,
                                       dtype=torch.int64)  # the batch-index has to be integer
        self.opposite_tensor = torch.tensor(self.lattice.stencil.opposite, device=self.lattice.device,
                                            dtype=torch.int64)  # batch-index has to be a tensor
        # f_collided = torch.zeros_like(self.f_index[:, 0], dtype=self.lattice.dtype)
        # f_collided_opposite = torch.zeros_like(self.f_index[:, 0], dtype=self.lattice.dtype)
        # self.f_collided = torch.stack((f_collided, f_collided_opposite), dim=1)

    def __call__(self, f):
        # calc force on boundary:
        if self.calc_force:
            self.calc_force_on_boundary()
        # bounce (invert populations on fluid nodes neighboring solid nodes)
        # f = torch.where(self.f_mask[self.lattice.stencil.opposite], f_collided[self.lattice.stencil.opposite], f)

        if self.lattice.D == 2:
            f[self.opposite_tensor[self.f_index[:, 0]],
              self.f_index[:, 1],
              self.f_index[:, 2]] = self.f_collided[:, 0]
        if self.lattice.D == 3:
            f[self.opposite_tensor[self.f_index[:, 0]],
              self.f_index[:, 1],
              self.f_index[:, 2],
              self.f_index[:, 3]] = self.f_collided[:, 0]
        return f

    def make_no_stream_mask(self, f_shape):
        assert self.mask.shape == f_shape[1:]  # all dimensions of f except the 0th (q)
        # no_stream_mask has to be dimensions: (q,x,y,z) (z optional), but CAN be (x,y,z) (z optional).
        # ...in the latter case, torch.where broadcasts the mask to (q,x,y,z), so ALL q populations of a lattice-node are marked equally
        return self.lattice.convert_to_tensor(self.mask)

    def make_no_collision_mask(self, f_shape):
        # INFO: for the halfway bounce back boundary, a no_collision_mask ist not necessary, because the no_stream_mask
        # ...prevents interaction between nodes inside and outside of the boundary region.
        # INFO: pay attention to the initialization of observable/moment-fields (u, rho,...) on the boundary nodes,
        # ...in the initial solution of your flow, especially if visualization or post processing uses the field-values
        # ...in the whole domain (including the boundary region)!
        assert self.mask.shape == f_shape[1:]
        return self.lattice.convert_to_tensor(self.mask)

    def calc_force_on_boundary(self):
        # calculate force on boundary by momentum exchange method (MEA, MEM) according to Kruger et al., 2017, pp.215-217:
        # momentum (f_i*c_i - f_i_opposite*c_i_opposite = 2*f_i*c_i for a resting boundary) is summed for all...
        # ...populations pointing at the surface of the boundary
        self.force_sum = 2 * torch.einsum('i..., id -> d', self.f_collided[:, 0], self.lattice.e[self.f_index[:, 0]])

    def store_f_collided(self, f_collided):
        if self.lattice.D == 2:
            self.f_collided[:, 0] = torch.clone(f_collided[self.f_index[:, 0],  # q
                                                          self.f_index[:, 1],  # x
                                                          self.f_index[:, 2]])  # y
            self.f_collided[:, 1] = torch.clone(f_collided[self.opposite_tensor[self.f_index[:,0]],  # q
                                                          self.f_index[:, 1],  # x
                                                          self.f_index[:, 2]])  # y
        if self.lattice.D == 3:
            self.f_collided[:, 0] = torch.clone(f_collided[self.f_index[:, 0],  # q
                                                          self.f_index[:, 1],  # x
                                                          self.f_index[:, 2],  # y
                                                          self.f_index[:, 3]])  # z
            self.f_collided[:, 1] = torch.clone(f_collided[self.opposite_tensor[self.f_index[:,0]],  # q
                                                          self.f_index[:, 1],  # x
                                                          self.f_index[:, 2],  # y
                                                          self.f_index[:, 3]])  # z

    def initialize_f_collided(self):
        f_collided = torch.zeros_like(self.f_index[:, 0], dtype=self.lattice.dtype)
        f_collided_opposite = torch.zeros_like(self.f_index[:, 0], dtype=self.lattice.dtype)
        self.f_collided = torch.stack((f_collided, f_collided_opposite), dim=1)

class HalfwayBounceBackBoundary_compact_v3:

    def __init__(self, mask, lattice):
        self.mask = mask
        self.lattice = lattice
        self.force_sum = torch.zeros_like(self.lattice.convert_to_tensor(
            self.lattice.stencil.e[0]))  # summed force vector on all boundary nodes, in D dimensions (x,y,(z))
        ### create f_mask, needed for force-calculation
        # ...(marks all fs which point from fluid to solid (boundary))

        self.f_index_fluid = []  # marks population from boundary-neighboring fluid node, pointing inside the boundary
        self.f_index_solid = []  # marks population from fluid-neighboring boundary node, pointing inside the boundary (stores f_collided for force calculation and bounce)

        if self.lattice.D == 2:
            nx, ny = mask.shape  # domain size in x and y
            self._no_stream_mask = np.zeros((self.lattice.Q, nx, ny), dtype=bool)
            self._no_stream_mask = self._no_stream_mask | self.mask
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
                            # ...the direction pointing from the fluid neighbour to solid p is marked on the neighbour
                            self.f_index_fluid.append([self.lattice.stencil.opposite[i],
                                                 a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                 b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny])
                            self.f_index_solid.append([self.lattice.stencil.opposite[i], a[p], b[p]])

                            self._no_stream_mask[self.lattice.stencil.opposite[i], a[p], b[p]] = False  # allows storage of bounce-relevant populations
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there
        if self.lattice.D == 3:  # like 2D, but in 3D...guess what...
            nx, ny, nz = mask.shape
            self._no_stream_mask = np.zeros((self.lattice.Q, nx, ny, nz), dtype=bool)
            self._no_stream_mask = self._no_stream_mask | self.mask
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
                            self.f_index_fluid.append([self.lattice.stencil.opposite[i],
                                                 a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                 b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                                 c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz])
                            self.f_index_solid.append([self.lattice.stencil.opposite[i], a[p], b[p], c[p]])

                            self._no_stream_mask[self.lattice.stencil.opposite[i], a[p], b[p], c[p]] = False  # allows storage of bounce-relevant populations
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there
        self.f_index_fluid = torch.tensor(np.array(self.f_index_fluid), device=self.lattice.device,
                                          dtype=torch.int64)  # the batch-index has to be integer
        self.f_index_solid = torch.tensor(np.array(self.f_index_solid), device=self.lattice.device,
                                          dtype=torch.int64)  # the batch-index has to be integer
        self.opposite_tensor = torch.tensor(self.lattice.stencil.opposite, device=self.lattice.device,
                                            dtype=torch.int64)  # batch-index has to be a tensor
        self.stencil_e_tensor_index = torch.tensor(self.lattice.e, device=self.lattice.device, dtype=torch.int64)


    def __call__(self, f):
        # bounce (invert populations on fluid nodes neighboring solid nodes)
        # f = torch.where(self.f_mask[self.lattice.stencil.opposite], f_collided[self.lattice.stencil.opposite], f)

        if self.lattice.D == 2:
            f[self.opposite_tensor[self.f_index_fluid[:, 0]],
              self.f_index_fluid[:, 1],
              self.f_index_fluid[:, 2]] = f[self.f_index_solid[:, 0],
                                            self.f_index_solid[:, 1],
                                            self.f_index_solid[:, 2]]
        if self.lattice.D == 3:
            f[self.opposite_tensor[self.f_index_fluid[:, 0]],
              self.f_index_fluid[:, 1],
              self.f_index_fluid[:, 2],
              self.f_index_fluid[:, 3]] = f[self.f_index_solid[:, 0],
                                            self.f_index_solid[:, 1],
                                            self.f_index_solid[:, 2],
                                            self.f_index_solid[:, 3]]

        # calc force on boundary:
        self.calc_force_on_boundary(f)
        return f

    def make_no_stream_mask(self, f_shape):
        assert self._no_stream_mask.shape == f_shape
        # no_stream_mask has to be dimensions: (q,x,y,z) (z optional), but CAN be (x,y,z) (z optional).
        # ...in the latter case, torch.where broadcasts the mask to (q,x,y,z), so ALL q populations of a lattice-node are marked equally
        return self.lattice.convert_to_tensor(self._no_stream_mask)

    def make_no_collision_mask(self, f_shape):
        # INFO: for the halfway bounce back boundary, a no_collision_mask ist not necessary, because the no_stream_mask
        # ...prevents interaction between nodes inside and outside of the boundary region.
        # INFO: pay attention to the initialization of observable/moment-fields (u, rho,...) on the boundary nodes,
        # ...in the initial solution of your flow, especially if visualization or post processing uses the field-values
        # ...in the whole domain (including the boundary region)!
        assert self.mask.shape == f_shape[1:]
        return self.lattice.convert_to_tensor(self.mask)

    def calc_force_on_boundary(self, f):
        if self.lattice.D == 2:
            # self.force_sum = 2 * torch.einsum('i..., id -> d', f[self.opposite_tensor[self.f_index_fluid[:, 0]],
            #                                                      self.f_index_fluid[:, 1],
            #                                                      self.f_index_fluid[:, 2]],
            #                                   self.lattice.e[self.f_index_fluid[:, 0]])
            # self.force_sum = 2 * torch.einsum('i..., id -> d',
            #                                   f[self.f_index_fluid[:, 0],
            #                                      self.f_index_fluid[:, 1] + self.stencil_e_tensor_index[self.f_index_fluid[:, 0], 0],
            #                                      self.f_index_fluid[:, 2] + self.stencil_e_tensor_index[self.f_index_fluid[:, 0], 1]],
            #                                   self.lattice.e[self.f_index_fluid[:, 0]])
            self.force_sum = 2 * torch.einsum('i..., id -> d', f[self.f_index_solid[:, 0],
                                                                 self.f_index_solid[:, 1],
                                                                 self.f_index_solid[:, 2]],
                                              self.lattice.e[self.f_index_solid[:, 0]])
        if self.lattice.D == 3:
            self.force_sum = 2 * torch.einsum('i..., id -> d', f[self.f_index_solid[:, 0],
                                                                 self.f_index_solid[:, 1],
                                                                 self.f_index_solid[:, 2],
                                                                 self.f_index_solid[:, 3]],
                                              self.lattice.e[self.f_index_solid[:, 0]])
            # self.force_sum = 2 * torch.einsum('i..., id -> d', f_collided.to_dense()[self.f_index_fluid[:, 0],
            #                                                                          self.f_index_fluid[:, 1],
            #                                                                          self.f_index_fluid[:, 2],
            #                                                                          self.f_index_fluid[:, 3]],
            #                                   self.lattice.e[self.f_index_fluid[:, 0]])
            # f_tmp = f[self.f_index_fluid[:, 0],
            #           self.f_index_fluid[:, 1],
            #           self.f_index_fluid[:, 2],
            #           self.f_index_fluid[:, 3]]

            # f_tmp = f[self.f_index_solid[:, 0],
            #           self.f_index_solid[:, 1],
            #           self.f_index_solid[:, 2],
            #           self.f_index_solid[:, 3]]
            # e_tmp = self.lattice.e[self.f_index_solid[:, 0]]
            # print("f_tmp shape:", f_tmp.size())
            # print("e_tmp shape:", e_tmp.size())
            # self.force_sum = 2 * torch.einsum('i..., id -> d',
            #                                   f_tmp,
            #                                   e_tmp)

            # f_tmp = f[self.opposite_tensor[self.f_index_fluid[:, 0]],
            #           self.f_index_fluid[:, 1],
            #           self.f_index_fluid[:, 2],
            #           self.f_index_fluid[:, 3]]
            # e_tmp = self.lattice.e[self.f_index_fluid[:, 0]]
            # print(f_tmp.size())
            # print(e_tmp.size())
            # self.force_sum = 2 * torch.einsum('i..., id -> d',f_tmp, e_tmp)

            # self.force_sum = 2 * torch.einsum('i..., id -> d',
            #                                   f[self.f_index_solid[:, 0],
            #                                     self.f_index_solid[:, 1],
            #                                     self.f_index_solid[:, 2],
            #                                     self.f_index_solid[:, 3]],
            #                                   self.lattice.e[self.f_index_solid[:, 0]])

        # HIER BRAUCHE ICH VIELLEICHT NOCH EIN MINUS VOR DER BERECHNUNG...


class EquilibriumBoundaryPU:
    """Sets distributions on this boundary to equilibrium with predefined velocity and pressure.
    Note that this behavior is generally not compatible with the Navier-Stokes equations.
    This boundary condition should only be used if no better options are available.
    """

    def __init__(self, mask, lattice, units, velocity, pressure=0):
        # parameter input (u, p) in PU!
        # u can be a field (individual ux, uy, (uz) for all boundary nodes) or vector (uniform ux, uy, (uz)))
        self.mask = lattice.convert_to_tensor(mask)
        self.lattice = lattice
        self.units = units
        self.velocity = lattice.convert_to_tensor(velocity)
        self.pressure = lattice.convert_to_tensor(pressure)

    def __call__(self, f):
        # convert PU-inputs to LU, calc feq and overwrite f with feq where mask==True
        rho = self.units.convert_pressure_pu_to_density_lu(self.pressure)
        u = self.units.convert_velocity_to_lu(self.velocity)
        feq = self.lattice.equilibrium(rho, u)
        feq = self.lattice.einsum("q,q->q", [feq, torch.ones_like(f)])
        f = torch.where(self.mask, feq, f)
        return f

    # TODO: add no_streaming_mask to counter the BC-order artifacts through "in"-streaming of outlet populations!

class AntiBounceBackOutlet:
    """Allows distributions to leave domain unobstructed through this boundary.
        Based on equations from page 195 of "The lattice Boltzmann method" (2016 by Krüger et al.)
        Give the side of the domain with the boundary as list [x, y, z] with only one entry nonzero
        [1, 0, 0] for positive x-direction in 3D; [1, 0] for the same in 2D
        [0, -1, 0] is negative y-direction in 3D; [0, -1] for the same in 2D
        """

    def __init__(self, lattice, direction):

        assert isinstance(direction, list), \
            LettuceException(
                f"Invalid direction parameter. Expected direction of type list but got {type(direction)}.")

        assert len(direction) in [1, 2, 3], \
            LettuceException(
                f"Invalid direction parameter. Expected direction of of length 1, 2 or 3 but got {len(direction)}.")

        assert (direction.count(0) == (len(direction) - 1)) and ((1 in direction) ^ (-1 in direction)), \
            LettuceException(
                "Invalid direction parameter. "
                f"Expected direction with all entries 0 except one 1 or -1 but got {direction}.")

        direction = np.array(direction)
        self.lattice = lattice

        # select velocities to be bounced (the ones pointing in "direction")
        self.velocities = np.concatenate(np.argwhere(np.matmul(self.lattice.stencil.e, direction) > 1 - 1e-6), axis=0)

        # build indices of u and f that determine the side of the domain
        self.index = []
        self.neighbor = []
        for i in direction:
            if i == 0:
                self.index.append(slice(None))
                self.neighbor.append(slice(None))
            if i == 1:
                self.index.append(-1)
                self.neighbor.append(-2)
            if i == -1:
                self.index.append(0)
                self.neighbor.append(1)
        # construct indices for einsum and get w in proper shape for the calculation in each dimension
        if len(direction) == 3:
            self.dims = 'dc, cxy -> dxy'
            self.w = self.lattice.w[self.velocities].view(1, -1).t().unsqueeze(1)
        if len(direction) == 2:
            self.dims = 'dc, cx -> dx'
            self.w = self.lattice.w[self.velocities].view(1, -1).t()
        if len(direction) == 1:
            self.dims = 'dc, c -> dc'
            self.w = self.lattice.w[self.velocities]

    def __call__(self, f):
        u = self.lattice.u(f)
        u_w = u[[slice(None)] + self.index] + 0.5 * (u[[slice(None)] + self.index] - u[[slice(None)] + self.neighbor])
        f[[np.array(self.lattice.stencil.opposite)[self.velocities]] + self.index] = (
                - f[[self.velocities] + self.index] + self.w * self.lattice.rho(f)[[slice(None)] + self.index] *
                (2 + torch.einsum(self.dims, self.lattice.e[self.velocities], u_w) ** 2 / self.lattice.cs ** 4
                 - (torch.norm(u_w, dim=0) / self.lattice.cs) ** 2)
        )
        return f

    def make_no_stream_mask(self, f_shape):
        no_stream_mask = torch.zeros(size=f_shape, dtype=torch.bool, device=self.lattice.device)
        no_stream_mask[[np.array(self.lattice.stencil.opposite)[self.velocities]] + self.index] = 1
        return no_stream_mask

    # not 100% sure about this. But collisions seem to stabilize the boundary.
    # def make_no_collision_mask(self, f_shape):
    #    no_collision_mask = torch.zeros(size=f_shape[1:], dtype=torch.bool, device=self.lattice.device)
    #    no_collision_mask[self.index] = 1
    #    return no_collision_mask


class EquilibriumOutletP(AntiBounceBackOutlet):
    """Equilibrium outlet with constant pressure.
    """

    def __init__(self, lattice, direction, rho0=1.0):
        super(EquilibriumOutletP, self).__init__(lattice, direction)
        self.rho0 = rho0

    def __call__(self, f):
        here = [slice(None)] + self.index
        other = [slice(None)] + self.neighbor
        rho = self.lattice.rho(f)
        u = self.lattice.u(f)
        rho_w = self.rho0 * torch.ones_like(rho[here])
        u_w = u[other]
        f[here] = self.lattice.equilibrium(rho_w[..., None], u_w[..., None])[..., 0]
        return f

    def make_no_stream_mask(self, f_shape):
        no_stream_mask = torch.zeros(size=f_shape, dtype=torch.bool, device=self.lattice.device)
        no_stream_mask[[np.setdiff1d(np.arange(self.lattice.Q), self.velocities)] + self.index] = 1
        return no_stream_mask

    def make_no_collision_mask(self, f_shape):
        no_collision_mask = torch.zeros(size=f_shape[1:], dtype=torch.bool, device=self.lattice.device)
        no_collision_mask[self.index] = 1
        return no_collision_mask
