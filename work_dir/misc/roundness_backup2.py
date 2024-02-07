import lettuce as lt
import matplotlib.pyplot as plt
import numpy as np
import torch

# List of Diemeters to measure:
gpds = [20,21,22] #(np.arange(150)+5)
r_rel_list = []
for i in gpds:
    gridpoints_per_diameter = i  # gp_per_D -> this defines the resolution ( D_LU = GPD+1)
    domain_width_in_D = 1+1/gridpoints_per_diameter  # D/Y  -> this defines the domain-size and total number of Lattice-Nodes
    domain_length_in_D = 1+1/gridpoints_per_diameter #2*domain_width_in_D  # D/X

    # if DpY is even, resulting GPD can't be odd for symmetrical cylinder and channel
    # ...if DpY is even, GPD will be corrected to even GPD for symemtrical cylinder
    # ...use odd DpY to use odd GPD
    gpd_correction=False
    if domain_width_in_D % 2 == 0 and gridpoints_per_diameter % 2 != 0:
        gpd_correction = True   # gpd_was_corrected-flag
        gpd_setup = gridpoints_per_diameter   # store old gpd for output
        gridpoints_per_diameter = int(gridpoints_per_diameter/2)*2   # make gpd even
        print("(!) domain_width_in_D is even, gridpoints_per_diameter will be "+str(gridpoints_per_diameter)+". Use odd domain_width_in_D to enable use of odd GPD!")

    print("shape_LU:", gridpoints_per_diameter*domain_length_in_D, "x", gridpoints_per_diameter*domain_width_in_D)
    ##gridpoints = gridpoints_per_diameter**2*domain_length_in_D*domain_width_in_D
    ##print("No. of gridpoints:", gridpoints)

    # lattice (for D, Q and e (stencil))
    lattice = lt.Lattice(lt.D2Q9, "cuda:0", dtype=torch.float64)

    # shape (x,y,z) of the domain
    shape = (int(domain_length_in_D * gridpoints_per_diameter), int(domain_width_in_D * gridpoints_per_diameter))

    # define radius and position for a symetrical circular Cylinder-Obstacle
    radius_LU = 0.5 * gridpoints_per_diameter
    y_pos_LU = 0.5 * gridpoints_per_diameter * domain_width_in_D + 0.5
    x_pos_LU = y_pos_LU

    # get x,y,z meshgrid of the domain (LU)
    xyz = tuple(np.linspace(1, n, n) for n in shape)  # Tupel aus Listen indizes (1-n (nicht 0-based!))
    xLU, yLU = np.meshgrid(*xyz, indexing='ij')  # meshgrid aus den x-, y- (und z-)Indizes -> * damit man die einzelnen Einträge des Tupels übergibt, und nicht das eine Tupel

    # define cylinder (LU)
    obstacle_mask = np.sqrt((xLU - x_pos_LU) ** 2 + (yLU - y_pos_LU) ** 2) < radius_LU

   # collect data on radii on the outside of the cylinder (all gridpoints within the cylinder, that have at least one stencil-link to a fluid node outside the cylinder)
    if lattice.D == 2:
        nx, ny = obstacle_mask.shape  # Anzahl x-Punkte, Anzahl y-Punkte (Skalar), (der gesamten Domain)

        rand_mask = np.zeros((nx, ny), dtype=bool)  # für Randpunkte, die es gibt
        rand_mask_f = np.zeros((lattice.Q, nx, ny), dtype=bool)  # für Randpunkte (inkl. Q-Dimension)
        rand_xq = []  # Liste aller x Werte (inkl. q-multiplizität)
        rand_yq = []  # Liste aller y Werte (inkl. q-multiplizität)

        a, b = np.where(obstacle_mask)  # np.array: Liste der (a) x-Koordinaten  und (b) y-Koordinaten der obstacle_mask
        # ...um über alle Boundary/Objekt/Wand-Knoten iterieren zu können
        for p in range(0, len(a)):  # für alle TRUE-Punkte der obstacle_mask
            for i in range(0, lattice.Q):  # für alle stencil-Richtungen c_i (hier lattice.stencil.e)
                try:  # try in case the neighboring cell does not exist (an f pointing out of the simulation domain)
                    if not obstacle_mask[a[p] + lattice.stencil.e[i, 0], b[p] + lattice.stencil.e[i, 1]]:
                        # falls in einer Richtung Punkt+(e_x, e_y; e ist c_i) False ist, ist das also ein Oberflächepunkt des Objekts (selbst True mit Nachbar False)
                        rand_mask[a[p], b[p]] = 1
                        rand_mask_f[lattice.stencil.opposite[i], a[p], b[p]] = 1
                        rand_xq = a[p]
                        rand_yq = b[p]
                except IndexError:
                    pass  # just ignore this iteration since there is no neighbor there
    if lattice.D == 3:  # entspricht 2D, nur halt in 3D...guess what...
        nx, ny, nz = obstacle_mask.shape

        rand_mask = np.zeros((nx, ny, nz), dtype=bool)
        rand_mask_f = np.zeros((lattice.Q, nx, ny, nz), dtype=bool)

        a, b, c = np.where(obstacle_mask)
        for p in range(0, len(a)):
            for i in range(0, lattice.Q):
                try:  # try in case the neighboring cell does not exist (an f pointing out of simulation domain)
                    if not obstacle_mask[a[p] + lattice.stencil.e[i, 0], b[p] + lattice.stencil.e[i, 1], c[p] + lattice.stencil.e[i, 2]]:
                        rand_mask[a[p], b[p], c[p]] = 1
                        rand_mask_f[lattice.stencil.opposite[i], a[p], b[p], c[p]] = 1
                except IndexError:
                    pass  # just ignore this iteration since there is no neighbor there

    rand_x, rand_y = np.where(rand_mask)  # Liste aller Rand-x- und y-Koordinaten
    x_pos = sum(rand_x)/len(rand_x)  # x_Koordinate des Kreis-Zentrums
    y_pos = sum(rand_y)/len(rand_y)  # y-Koordiante des Kreis-Zentrums

    radii_q = np.sqrt(np.power(np.array(rand_xq)-rand_x, 2) + np.power(np.array(rand_yq)-rand_y, 2))  # Liste der Radien in LU (multiplizität, falls ein Randpunkt mehrere Links zu Fluidknoten hat

    r_max = 0
    r_min = gridpoints_per_diameter
    radii = np.zeros_like(rand_x)  # Liste aller Radien (ohne q) in LU
    for p in range(0, len(rand_x)):  # für alle Punkte
        radii[p] = np.sqrt((rand_x[p]-x_pos)**2 +(rand_y[p]-y_pos)**2)  # berechne Abstand des Punktes zum Zentrum
        if radii[p] > r_max:
            r_max = radii[p]
        if radii[p] < r_min:
            r_min = radii[p]

    radii_relative = radii / r_max
    radii_q_relative = radii_q / r_max

    r_rel_list.append(radii_relative)

    from collections import Counter
    print("GPD: ", gridpoints_per_diameter)
    print(Counter(radii))


r_rel_list = np.array(r_rel_list)
n, bins, patches = plt.hist(x=radii_relative,  #bins='auto',
                            color='#0504aa',
                            alpha=0.7,
                            rwidth=0.85,  #density=True,
                            align="left",
                            weights=np.ones_like(radii_relative) / len(radii_relative)  # y-Achse
                            )
plt.grid(axis='y', alpha=0.75)
plt.xlabel('rel. radius')
plt.ylabel('Frequency')
plt.title('Histogram of relative radius distribution for '+str(gridpoints_per_diameter)+" GPD")
plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
#plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
#plt.show()
#plt.savefig("/home/max/Desktop/roundness_histograms/Histogram_GPD" + str(gridpoints_per_diameter))
plt.show()

plt.figure()
#plt.imshow(obstacle_mask)
plt.imshow(rand_mask)
plt.show()
#plt.savefig("/home/max/Desktop/GPD_shapes/GPD_Histogram"+str(gridpoints_per_diameter))
pass