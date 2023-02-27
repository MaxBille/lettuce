import torch
import numpy as np


class ForceOnBoundary:
    """calculates the force on the boundary area defined by boundary_mask by MEA/MEM (momentum exchange algorithm),
    based on code by M.Kliemank, who based his code on Kruger (book, 2007, "principles and pracites...").
    - lettuce implemented a Fullway Bounce-Back Boundary, which could be problematic, Kruger describes the MEM for
    a halfway-bbb, which is more time-accurate
    """
    def __init__(self, boundary_mask, lattice):
        print("initializing forceOnBoundary")
        self.boundary_mask = lattice.convert_to_tensor(boundary_mask)
        self.lattice = lattice

        self.force = torch.zeros_like(self.lattice.convert_to_tensor(self.lattice.stencil.e[0]))  # for force in all dimensions (x,y,(z))
        if lattice.D == 2:
            x, y = boundary_mask.shape  # Anzahl x-Punkte, Anzahl y-Punkte (Skalar), der Simulation
            self.force_mask = np.zeros((lattice.Q, x, y), dtype=bool)  # force_mask: [stencilVektor-Zahl, x, y]
                # ...zur markierung aller auf die Boundary (bzw. das Objekt) zeigenden Stencil-Vektoren
            a, b = np.where(boundary_mask)  # np.array: Liste der (a) x-Koordinaten  und (b) y-Koordinaten der boundary-mask
                # ...um über alle Boundary-Knoten iterieren zu können
            for p in range(0, len(a)):  # für alle TRUE-Punkte der boundary-mask
                for i in range(0, lattice.Q):  # für alle stencil-Richtungen c_i (hier lattice.stencil.e)
                    try:  # try in case the neighboring cell does not exist (an f pointing out of the simulation domain)
                        if not boundary_mask[a[p] + lattice.stencil.e[i, 0], b[p] + lattice.stencil.e[i, 1]]:
                            # falls in einer Richtung Punkt+(e_x, e_y; e ist c_i) False ist, ist das also ein Oberflächepunkt des Objekts (selbst True mit Nachbar False)
                            # ...wird der gegenüberliegende stencil-Vektor e_i, des nach außen zeigenden Stencil-Vektors (also letztendlich der in Richtung boundary zeigende Vektor)
                            # ...markiert:
                            # Markiere c_i (e) auf dem Boundary-Rand, welcher nach innen zeigt (vom ersten solid Knoten (auf dem Knoten selbst) aus, in Richtung Objektinneres)
                            # ...das ist die Population, die in diesem Zeitschritt in die Boundary eingeströmt ist und
                            # ...die von der Boundary in diesem Zeitschritt invertiert wird, also konkret die Population dieses Knotens, deren Impuls relevant ist
                            # (Schritt-Reihenfolge: Collision->Streaming->ForceCalc->Boundary, das heißt, "hier" wurde noch nicht invertiert)
                            self.force_mask[self.lattice.stencil.opposite[i], a[p], b[p]] = 1  # markiere alle gegen die Boundary gerichteten Populationen
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there
        if lattice.D == 3:  # entspricht 2D, nur halt in 3D...
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

        print("done initializing forceOnBoundary")
        print("force_mask before array-tensor-conversion:")
        print(self.force_mask)
        self.force_mask = self.lattice.convert_to_tensor(self.force_mask)  # np.array to torch.tensor conversion
        print("force_mask after array-tensor-conversion:")
        print(self.force_mask)

    def __call__(self, f):
        print("calling forceOnBoundary")
        tmp = torch.where(self.force_mask, f, torch.zeros_like(f))  # alle Populationen f, welche auf dem Boundaryrand (solid) nach innen zeigen und im Boundary-Teilschritt invertiert werden würden
        self.force = 1 ** self.lattice.D * 2 * torch.einsum('i..., id -> d', tmp, self.lattice.e) / 1.0  # BERECHNET KRAFT - warum 1^D?...
            # summiert alle Kräfte in x und in y Richtung auf,
            # tmp: Betrag aus f an allen Stellen, die in force_mask markiert sind
            # Vorzeichen kommt über die Koordianten der Stencil-Einheitsvektoren (e[0 bis 8])
            # tmp: 9 x nx x ny
            # self.lattice.e: 9 x 2 (2D) bzw. 9 x 3 (3D)
            # Zuordnung der Multiplikation über die 9 Einheitsvektoren (Richtungen, indexname i)
            # übrig bleiben nur zwei (drei) Koordinatenrichtungen (indexname d)
            # "1**self-lattice.D" = dx³ (3D) bzw. dx² (2D) als Vorfaktor, welcher einheitenmäßig aus Impulsdichte einen Impuls macht
                # eigentlich rechnet man hier einen DELTA P aus
                # unter Annahme des stetigen Impulsaustauschs über dt, kann die Kraft als F= dP/dt berechnet werden
                # ...deshalb wird hier nochmal durch 1.0 geteilt (!)
        print("tmp: \n", tmp)
        print("self.lattice.e \n", self.lattice.e)
        print("force: \n", self.force)
            ### >>> stuff war schon bei M.K. auskommentiert:
            #tmp = torch.einsum("i..., id -> d...", tmp, self.lattice.e)
            #for _ in range(0, self.lattice.D):
            #    tmp =torch.sum(tmp, dim=1)
            # self.force = tmp * 2
            ### <<<
        return self.force  # force in x and y direction
