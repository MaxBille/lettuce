import torch
import numpy as np


class ForceOnBoundary:
    """Fullway Bounce-Back Boundary"""
    def __init__(self, boundary_mask, lattice):
        self.boundary_mask = lattice.convert_to_tensor(boundary_mask)
        self.lattice = lattice

        self.force = torch.zeros_like(self.lattice.convert_to_tensor(self.lattice.stencil.e[0]))
        if lattice.D == 2:
            x, y = boundary_mask.shape # Anzahl x-Punkte, Anzahl y-Punkte (Skalar)
            print("x:", x)
            print("y:", y)
            self.force_mask = np.zeros((lattice.Q, x, y), dtype=bool) # force_mask: [stencilVektor-Zahl, x, y]
            a, b = np.where(boundary_mask) # np.array, welches
            print("a:", a) # Liste der x-Koordinaten der boundary-mask
            print("b:", b) # Liste der y-Koordinaten der boudnary-mask
            for p in range(0, len(a)): # für alle Punkte der boundary-mask
                for i in range(0, lattice.Q): # für alle stencil-Richtungen c_i (hier lattice.stencil.e)
                    try:  # try in case the neighboring cell does not exist (an f pointing out of simulation domain)
                        if not boundary_mask[a[p] + lattice.stencil.e[i, 0], b[p] + lattice.stencil.e[i, 1]]:
                            # falls in einer Richtung Punkt+(e_x, e_y; e ist c_i) False ist, ist das also ein Oberflächepunkt des Objekts (True mit Nachbar False)
                            # ...wird der gegenüberliegende stencil-Vektor e_i, des nach außen zeigenden Stencil-Vektors (also letztendlich der in Richtung boundary zeigende)
                            # ...markiert:
                            # Markiere c_i auf dem Boundary-Rand, welcher nach innen zeigt (vom solid Knoten aus)
                            self.force_mask[self.lattice.stencil.opposite[i], a[p], b[p]] = 1
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there
        if lattice.D == 3:
            x, y, z = boundary_mask.shape
            self.force_mask = np.zeros((lattice.Q, x, y, z), dtype=bool)
            a, b, c = np.where(boundary_mask)
            for p in range(0, len(a)):
                for i in range(0, lattice.Q):
                    try:  # try in case the neighboring cell does not exist (an f pointing out of simulation domain)
                        if not boundary_mask[a[p] + lattice.stencil.e[i, 0], b[p] + lattice.stencil.e[i, 1], c[p] + lattice.stencil.e[i, 2]]:
                            self.force_mask[self.lattice.stencil.opposite[i], a[p], b[p], c[p]] = 1
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there

        self.force_mask = self.lattice.convert_to_tensor(self.force_mask)
        print(self.force_mask)

    def __call__(self, f):
        tmp = torch.where(self.force_mask, f, torch.zeros_like(f)) # x,y-Grid mit f an den Stellen, wo ein Boundary-Rand ist (auf der Boundary)
        self.force = 1 ** self.lattice.D * 2 * torch.einsum('i..., id -> d', tmp, self.lattice.e) / 1.0 # warum 1^D?
        #tmp = torch.einsum("i..., id -> d...", tmp, self.lattice.e)
        #for _ in range(0, self.lattice.D):
        #    tmp = torch.sum(tmp, dim=1)
        # self.force = tmp * 2
        return self.force
