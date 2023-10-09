import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

__all__ = ["draw_circular_mask"]

def draw_circular_mask(lattice, gridpoints_per_diameter, output_data=False, filebase=".", print_data=False):
    ### calculate and export 2D obstacle_mask as .png
    grid_x = gridpoints_per_diameter + 2
    if output_data:
        output_file = open(filebase + "/obstacle_mask_info.txt", "a")
        output_file.write("GPD = " + str(gridpoints_per_diameter) + "\n")
    if print_data:
        print("GPD = " + str(gridpoints_per_diameter))
    # define radius and position for a symetrical circular Cylinder-Obstacle
    radius_LU = 0.5 * gridpoints_per_diameter
    y_pos_LU = 0.5 * grid_x + 0.5
    x_pos_LU = y_pos_LU

    # get x,y,z meshgrid of the domain (LU)
    xyz = tuple(np.linspace(1, n, n) for n in (grid_x, grid_x))  # Tupel aus Listen indizes (1-n (nicht 0-based!))
    xLU, yLU = np.meshgrid(*xyz, indexing='ij')  # meshgrid aus den x-, y- (und z-)Indizes -> * damit man die einzelnen Einträge des Tupels übergibt, und nicht das eine Tupel

    # define cylinder (LU) (circle)
    obstacle_mast_for_visualization = np.sqrt((xLU - x_pos_LU) ** 2 + (yLU - y_pos_LU) ** 2) < radius_LU

    nx, ny = obstacle_mast_for_visualization.shape  # Anzahl x-Punkte, Anzahl y-Punkte (Skalar), (der gesamten Domain)

    rand_mask = np.zeros((nx, ny), dtype=bool)  # für Randpunkte, die es gibt
    rand_mask_f = np.zeros((lattice.Q, nx, ny), dtype=bool)  # für Randpunkte (inkl. Q-Dimension)
    rand_xq = []  # Liste aller x Werte (inkl. q-multiplizität)
    rand_yq = []  # Liste aller y Werte (inkl. q-multiplizität)

    a, b = np.where(obstacle_mast_for_visualization)  # np.array: Liste der (a) x-Koordinaten  und (b) y-Koordinaten der obstacle_mast_for_visualization
    # ...um über alle Boundary/Objekt/Wand-Knoten iterieren zu können
    for p in range(0, len(a)):  # für alle TRUE-Punkte der obstacle_mast_for_visualization
        for i in range(0, lattice.Q):  # für alle stencil-Richtungen c_i (hier lattice.stencil.e)
            try:  # try in case the neighboring cell does not exist (an f pointing out of the simulation domain)
                if not obstacle_mast_for_visualization[a[p] + lattice.stencil.e[i, 0], b[p] + lattice.stencil.e[i, 1]]:
                    # falls in einer Richtung Punkt+(e_x, e_y; e ist c_i) False ist, ist das also ein Oberflächenpunkt des Objekts (selbst True mit Nachbar False)
                    rand_mask[a[p], b[p]] = 1
                    rand_mask_f[lattice.stencil.opposite[i], a[p], b[p]] = 1
                    rand_xq.append(a[p])
                    rand_yq.append(b[p])
            except IndexError:
                pass  # just ignore this iteration since there is no neighbor there
    rand_x, rand_y = np.where(rand_mask)  # Liste aller Rand-x- und y-Koordinaten
    x_pos = sum(rand_x) / len(rand_x)  # x_Koordinate des Kreis-Zentrums
    y_pos = sum(rand_y) / len(rand_y)  # y-Koordinate des Kreis-Zentrums

    # calculate all radii and r_max and r_min
    r_max = 0
    r_min = gridpoints_per_diameter
    radii = np.zeros_like(rand_x, dtype=float)  # Liste aller Radien (ohne q) in LU
    for p in range(0, len(rand_x)):  # für alle Punkte
        radii[p] = np.sqrt(
            (rand_x[p] - x_pos) ** 2 + (rand_y[p] - y_pos) ** 2)  # berechne Abstand des Punktes zum Zentrum
        if radii[p] > r_max:
            r_max = radii[p]
        if radii[p] < r_min:
            r_min = radii[p]

    # calculate all radii (with q-multiplicity)
    radii_q = np.zeros_like(rand_xq, dtype=float)
    for p in range(0, len(rand_xq)):
        radii_q[p] = np.sqrt((rand_xq[p] - x_pos) ** 2 + (rand_yq[p] - y_pos) ** 2)

    ### all relative radii in relation to gpd/2
    radii_relative = radii / (
                radius_LU - 0.5)  # (substract 0.5 because "true" boundary location is 0.5LU further out than node-coordinates)
    radii_q_relative = radii_q / (radius_LU - 0.5)

    # calc. mean rel_radius
    r_rel_mean = sum(radii_relative) / len(radii_relative)
    rq_rel_mean = sum(radii_q_relative) / len(radii_q_relative)

    ## AREA calculation
    area_theory = np.pi * (gridpoints_per_diameter / 2) ** 2  # area = pi*r² in LU²
    area = len(a)  # area in LU = number of nodes, because every node has a cell of 1LU x 1LU around it

    if output_data:
        output_file.write("\nr_rel_mean: " + str(sum(radii_relative) / len(radii_relative)))
        output_file.write("\nrq_rel_mean: " + str(sum(radii_q_relative) / len(radii_q_relative)))
        output_file.write("\nr_rel_min: " + str(r_max / (radius_LU - 0.5)))
        output_file.write("\nr_rel_max: " + str(r_min / (radius_LU - 0.5)))
        output_file.write("\n\narea_rel: " + str(area / area_theory))

        output_file.write("\n\nradii: " + str(Counter(radii)))
        output_file.write("\nradii_q: " + str(Counter(radii_q)) + "\n\n")
        output_file.close()
    if print_data:
        print("area_rel: " + str(area / area_theory))

    ### PLOT Mask
    plt.figure()
    plt.imshow(obstacle_mast_for_visualization)
    # plt.xticks(np.arange(gridpoints_per_diameter + 2), minor=True)
    # plt.yticks(np.arange(gridpoints_per_diameter + 2), minor=True)
    ax = plt.gca()
    xmin, xmax = ax.get_xlim()
    ymax, ymin = ax.get_ylim()
    if gridpoints_per_diameter >= 10:
        plt.xticks(np.arange(0, xmax, int(xmax / 10)))
        plt.yticks(np.arange(0, ymax, int(ymax / 10)))
    else:
        plt.xticks(np.arange(0, xmax, 1))
        plt.yticks(np.arange(0, ymax, 1))
    plt.title("GPD = " + str(gridpoints_per_diameter))
    ax.set_xticks(np.arange(-.5, xmax, 1), minor=True)
    ax.set_yticks(np.arange(-.5, ymax, 1), minor=True)

    # grid thickness, cicrle, node marker
    x, y = np.meshgrid(np.linspace(0, int(xmax), int(xmax + 1)), np.linspace(0, int(ymax), int(ymax + 1)))
    if gridpoints_per_diameter < 30:
        ax.grid(which="minor", color="k", axis='both', linestyle='-', linewidth=2)
        circle = plt.Circle((xmax / 2 - 0.25, ymax / 2 - 0.25), gridpoints_per_diameter / 2, color='r', fill=False,
                            linewidth=1)
        ax.add_patch(circle)
        plt.plot(x, y, marker='.', linestyle='', color="b", markersize=1)
    elif gridpoints_per_diameter < 70:
        ax.grid(which="minor", color="k", axis='both', linestyle='-', linewidth=1)
        circle = plt.Circle((xmax / 2 - 0.25, ymax / 2 - 0.25), gridpoints_per_diameter / 2, color='r', fill=False,
                            linewidth=0.5)
        ax.add_patch(circle)
    elif gridpoints_per_diameter < 100:
        ax.grid(which="minor", color="k", axis='both', linestyle='-', linewidth=0.5)
    elif gridpoints_per_diameter < 150:
        ax.grid(which="minor", color="k", axis='both', linestyle='-', linewidth=0.25)

    if output_data:
        plt.savefig(filebase + "/obstacle_mask_GPD" + str(gridpoints_per_diameter) + ".png")
    if print_data:
        plt.show()
    else:
        plt.close()