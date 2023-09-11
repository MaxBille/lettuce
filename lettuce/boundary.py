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
from lettuce import (LettuceException)

__all__ = ["BounceBackBoundary", "HalfwayBounceBackBoundary", "FullwayBounceBackBoundary",
           "AntiBounceBackOutlet", "EquilibriumBoundaryPU", "EquilibriumOutletP", "SlipBoundary", "InterpolatedBounceBackBoundary"]

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
            nx, ny = mask.shape  # domain size in x and y
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
                        if not mask[a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
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
                            # distance (LU) from fluid node to the "true" boundary location
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
            nx, ny, nz = mask.shape
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
                            # distance (LU) from fluid node to the "true" boundary location
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
        print("IBB initialization took "+str(time.time()-t_init_start)+"seconds")

    def __call__(self, f, f_collided):

        if self.interpolation_order == 2:
            print("warning: not implemented")
        else:  #interpolation_order==1:
            # f_tmp = f_collided[i,x_b]_interpolation before bounce
            f_tmp = torch.where(self.d <= 0.5,  # if d<=1/2
                                2*self.d*f_collided+(1-2*self.d)*f,  # interpolate from second fluid node
                                (1/(2*self.d))*f_collided+(1-1/(2*self.d))*f_collided[self.lattice.stencil.opposite])  # else: interpolate from opposing populations on x_b
            # (?) 1-1/(2d) ODER (2d-1)/2d, welches ist numerisch exakter?
            # f_collided an x_f entspricht f_streamed an x_b, weil entlang des links ohne collision gestreamt wird!
            # ... d.h. f_collided[i,x_f] entspricht f[i,x_b]
            f = torch.where(self.f_mask[self.lattice.stencil.opposite], f_tmp[self.lattice.stencil.opposite], f)
            #HWBB: f = torch.where(self.f_mask[self.lattice.stencil.opposite], f_collided[self.lattice.stencil.opposite], f)

        self.calc_force_on_boundary(f, f_collided)
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

    def calc_force_on_boundary(self, f_bounced, f_collided):
        # momentum exchange according to Bouzidi et al. (2001), equation 11.8 in Kruger et al. (2017) p.445 // watch out for wrong signs. Kruger states "-", but "+" gives correct result
        # ...original paper (Bouzidi 2001) and Ginzburg followup (2003) state a "+" as well...
        #tmp = torch.where(self.f_mask, f_collided, torch.zeros_like(f_bounced)) \
        #      - torch.where(self.f_mask, f_bounced[self.lattice.stencil.opposite], torch.zeros_like(f_bounced))
        #tmp = torch.where(self.f_mask, f_collided - f_bounced[self.lattice.stencil.opposite], torch.zeros_like(f_bounced)) #WRONG
        tmp = torch.where(self.f_mask, f_collided + f_bounced[self.lattice.stencil.opposite], torch.zeros_like(f_bounced))  #RIGHT
        self.force_sum = torch.einsum('i..., id -> d', tmp, self.lattice.e)  # CALCULATE FORCE / v3.0 - M.Bille: dx_lu = dt_lu is allways 1 (!)

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
                    elif a[p] == nx-1 and self.lattice.e[i, 0] == 1:  # searching border on right
                        border[0] = 1
                    if b[p] == 0 and self.lattice.stencil.e[i, 1] == -1:  # searching border on left
                        border[1] = -1
                    elif b[p] == ny - 1 and self.lattice.e[i, 1] == 1:  # searching border on right
                        border[1] = 1
                    if c[p] == 0 and self.lattice.stencil.e[i, 2] == -1:  # searching border on left
                        border[2] = -1
                    elif c[p] == nz-1 and self.lattice.e[i, 2] == 1:  # searching border on right
                        border[2] = 1
                    try:  # try in case the neighboring cell does not exist (an f pointing out of simulation domain)
                        #if not mask[a[p] + self.lattice.stencil.e[i, 0]*single_layer[p, 0] - border_direction[p, 0]*nx,
                        #            b[p] + self.lattice.stencil.e[i, 1]*single_layer[p, 1] - border_direction[p, 1]*ny,
                        #            c[p] + self.lattice.stencil.e[i, 2]*single_layer[p, 2] - border_direction[p, 2]*nz]:
                        if not mask[a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                    b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                    c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz]:
                            #OLD: self.f_mask[self.lattice.stencil.opposite[i], a[p] + self.lattice.stencil.e[i, 0], b[p] + self.lattice.stencil.e[i, 1], c[p] + self.lattice.stencil.e[i, 2]] = 1
                            self.f_mask[self.lattice.stencil.opposite[i], a[p], b[p], c[p]] = 1
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there
#                if (b[p] == int(ny / 2)):
#                    print("node ({},{}): m={}, bdz={}, slz={}".format(str(a[p]), str(c[p]), mask[a[p], b[p], c[p]], border_direction[p, 2], single_layer[p, 2]))
#            print("border_direction\n", border_direction)
#            print("single_layer\n", single_layer)
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

class HalfwayBounceBackBoundary:
    """Halfway Bounce Back Boundary (with added force_on_boundary calculation)
    - halfway = inverts populations within one substep
    - call() must be called after Streaming substep
    - calculates the force on the boundary:
        - calculation is done after streaming, but theoretically the force is evaluated based on the populations touching/crossing the boundary IN this streaming step
    """

    # ToDo: self.mask und mask braucht man nicht beide, weil eigentlich f_mask genutzt wird!
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

    def __call__(self, f, f_collided):
        # HALFWAY-BB: overwrite all populations (on fluid nodes) which came from boundary with pre-streaming populations (on fluid nodes) which pointed at boundary
            #print("f_mask:\n", self.f_mask)
            #print("f_mask(q2,x1,y1):\n", self.f_mask[2, 1, 1])
            #print("f_mask(q2,x1,y3):\n", self.f_mask[2, 1, 3])
            #print("f_mask(opposite):\n", self.f_mask[self.lattice.stencil.opposite])
        # calc force on boundary:
        self.calc_force_on_boundary(f_collided)
        # bounce (invert populations on fluid nodes neighboring solid nodes)
        f = torch.where(self.f_mask[self.lattice.stencil.opposite], f_collided[self.lattice.stencil.opposite], f)
            # ersetze alle "von der boundary kommenden" Populationen durch ihre post-collision_pre-streaming entgegengesetzten Populationen
            # ...bounce-t die post_collision/pre-streaming Populationen an der Boundary innerhalb eines Zeitschrittes
            # ...von außen betrachtet wird "während des streamings", innerhalb des gleichen Zeitschritts invertiert.
            # (?) es wird keine no_streaming_mask benötigt, da sowieso alles, was aus der boundary geströmt käme hier durch pre-Streaming Populationen überschrieben wird.
            # ...ist das so, oder entsteht dadurch "Strömung" innerhalb des Obstacles? Diese hat zwar keinen direkten Einfluss auf die Größen im Fluidbereich,
            # ... lässt aber in der Visualisierung Werte ungleich Null innerhalb von Objekten entstehen und Mittelwerte etc. könnten davon beeinflusst werden. (?)
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
        #     self.force = 2 * torch.einsum('qxy, qd -> xyd', tmp,
        #                                   self.lattice.e)  # force = [x-coordinate, y-coodrinate, direction (0=x, 1=y)]
        # if self.lattice.D == 3:
        #     self.force = 2 * torch.einsum('qxyz, qd -> xyzd', tmp, self.lattice.e)  # force = [x-coordinate, y-coodrinate, z-coodrinate, direction (0=x, 1=y, 2=z)]


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
