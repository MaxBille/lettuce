import lettuce as lt
import matplotlib.pyplot as plt
import numpy as np
import torch

# List of Diameters to measure:
gpds = [9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,30,35,40,45,50,60,70,71,72,73,74,75,80,90,100] #(np.arange(150)+5)
gpds = [20,21,22]
#gpds = np.arange(0,150)+5
r_rel_list = []
r_rel_list_weights = []
rq_rel_list = []
rq_rel_list_weights = []

r_rel_mean_list = []
rq_rel_mean_list = []

r_rel_mean_PU_list = []
rq_rel_mean_PU_list = []

r_rel_max_list = []
r_rel_min_list = []

for i in gpds:
    print("#######")
    gridpoints_per_diameter = i  # gp_per_D -> this defines the resolution ( D_LU = GPD+1)
    domain_width_in_D = 1+(2/gridpoints_per_diameter)  # D/Y 19  -> this defines the domain-size and total number of Lattice-Nodes
    domain_length_in_D = 1+(2/gridpoints_per_diameter) #2*domain_width_in_D 2*domain_width_in_D # D/X

    # if DpY is even, resulting GPD can't be odd for symmetrical cylinder and channel
    # ...if DpY is even, GPD will be corrected to even GPD for symemtrical cylinder
    # ...use odd DpY to use odd GPD
    gpd_correction=False
    if False:#domain_width_in_D % 2 == 0 and gridpoints_per_diameter % 2 != 0:
        gpd_correction = True   # gpd_was_corrected-flag
        gpd_setup = gridpoints_per_diameter   # store old gpd for output
        gridpoints_per_diameter = int(gridpoints_per_diameter/2)*2   # make gpd even
        print("(!) domain_width_in_D is even, gridpoints_per_diameter will be "+str(gridpoints_per_diameter)+". Use odd domain_width_in_D to enable use of odd GPD!")

    print("shape_LU:", round(gridpoints_per_diameter*domain_length_in_D), "x", round(gridpoints_per_diameter*domain_width_in_D))
    ##gridpoints = gridpoints_per_diameter**2*domain_length_in_D*domain_width_in_D
    ##print("No. of gridpoints:", gridpoints)

    # lattice (for D, Q and e (stencil))
    lattice = lt.Lattice(lt.D2Q9, "cuda:0", dtype=torch.float64)

    # shape (x,y,z) of the domain
    shape = (round(domain_length_in_D * gridpoints_per_diameter), round(domain_width_in_D * gridpoints_per_diameter))

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
                        rand_xq.append(a[p])
                        rand_yq.append(b[p])
                except IndexError:
                    pass  # just ignore this iteration since there is no neighbor there
    # if lattice.D == 3:  # entspricht 2D, nur halt in 3D...guess what...
    #     nx, ny, nz = obstacle_mask.shape
    #
    #     rand_mask = np.zeros((nx, ny, nz), dtype=bool)
    #     rand_mask_f = np.zeros((lattice.Q, nx, ny, nz), dtype=bool)
    #
    #     a, b, c = np.where(obstacle_mask)
    #     for p in range(0, len(a)):
    #         for i in range(0, lattice.Q):
    #             try:  # try in case the neighboring cell does not exist (an f pointing out of simulation domain)
    #                 if not obstacle_mask[a[p] + lattice.stencil.e[i, 0], b[p] + lattice.stencil.e[i, 1], c[p] + lattice.stencil.e[i, 2]]:
    #                     rand_mask[a[p], b[p], c[p]] = 1
    #                     rand_mask_f[lattice.stencil.opposite[i], a[p], b[p], c[p]] = 1
    #             except IndexError:
    #                 pass  # just ignore this iteration since there is no neighbor there

    rand_x, rand_y = np.where(rand_mask)  # Liste aller Rand-x- und y-Koordinaten
    x_pos = sum(rand_x)/len(rand_x)  # x_Koordinate des Kreis-Zentrums
    y_pos = sum(rand_y)/len(rand_y)  # y-Koordiante des Kreis-Zentrums

    #radii_q = np.sqrt(np.power(np.array(rand_xq)-x_pos, 2) + np.power(np.array(rand_yq)-y_pos, 2))  # Liste der Radien in LU (multiplizität, falls ein Randpunkt mehrere Links zu Fluidknoten hat

    # calculate all radii and r_max and r_min
    r_max = 0
    r_min = gridpoints_per_diameter
    radii = np.zeros_like(rand_x, dtype=float)  # Liste aller Radien (ohne q) in LU
    for p in range(0, len(rand_x)):  # für alle Punkte
        radii[p] = np.sqrt((rand_x[p]-x_pos)**2 + (rand_y[p]-y_pos)**2)  # berechne Abstand des Punktes zum Zentrum
        if radii[p] > r_max:
            r_max = radii[p]
        if radii[p] < r_min:
            r_min = radii[p]

    # calculate all radii (with q-multiplicity)
    radii_q = np.zeros_like(rand_xq, dtype=float)
    for p in range(0, len(rand_xq)):
        radii_q[p] = np.sqrt((rand_xq[p]-x_pos)**2 + (rand_yq[p]-y_pos)**2)

    ### all relative radii in relation to gpd/2
    radii_relative = radii / radius_LU
    radii_q_relative = radii_q / radius_LU

    r_rel_list.append(radii_relative)
    r_rel_list_weights.append(np.ones_like(radii_relative) / len(radii_relative))
    rq_rel_list.append(radii_q_relative)
    rq_rel_list_weights.append(np.ones_like(radii_q_relative) / len(radii_q_relative))

    # mean radius
    r_rel_mean = sum(radii_relative)/len(radii_relative)
    rq_rel_mean_q = sum(radii_q_relative)/len(radii_q_relative)

    r_rel_mean_list.append(r_rel_mean)
    rq_rel_mean_list.append(rq_rel_mean_q)

    ### all relative radii in relation to D/2 in PU (D=GPD+1 (!))
    radii_relative_PU = (radii + 0.5) / (radius_LU + 0.5)
    radii_q_relative_PU = (radii_q + 0.5) / (radius_LU + 0.5)

    # mean radius in PU
    r_rel_mean_PU = sum(radii_relative_PU) / len(radii_relative_PU)
    rq_rel_mean_q_PU = sum(radii_q_relative_PU) / len(radii_q_relative_PU)

    r_rel_mean_PU_list.append(r_rel_mean_PU)
    rq_rel_mean_PU_list.append(rq_rel_mean_q_PU)

    # append max/min radii
    r_rel_max_list.append(r_max/radius_LU)
    r_rel_min_list.append(r_min/radius_LU)

    from collections import Counter
    print("GPD: ", gridpoints_per_diameter)
    print("radii: ", Counter(radii))
    print("radii_q: ", Counter(radii))

    if True:
        ### PLOT rand_mask
        plt.figure()
        plt.imshow(rand_mask)
        plt.xticks(np.arange(gridpoints_per_diameter + 2), minor=True)
        plt.yticks(np.arange(gridpoints_per_diameter + 2), minor=True)
        plt.xticks([])
        plt.yticks([])
        plt.title("GPD = "+str(gridpoints_per_diameter))
        plt.savefig("/home/max/Desktop/plots/roundness/other_masks/maskGPD" + str(gridpoints_per_diameter) + ".png")
        plt.show()

if False:
    ### HISTOGRAM for radii
    n, bins, patches = plt.hist(x=r_rel_list,  bins=list(np.linspace(0.86,1.0,int(0.15/0.01))),#bins='auto',
                                #color='#0504aa',
                                alpha=0.7,
                                rwidth=0.85,  #density=True,  # density macht nur % draus, durch *100
                                align="mid",
                                weights=r_rel_list_weights  # y-Achse
                                )
    plt.xlabel('relativer Radius')
    plt.ylabel('relative Häufigkeit')
    plt.title(
        'Histogram der relativen Häufigkeit realtiver Radien für verschiedene Auflösungen (GPD)',
        wrap=True)
    # plt.text(23, 45, r'$\mu=15, b=3$')
    plt.legend([str(x) + " GPD" for x in gpds])
    plt.ylim([0, 1])
    plt.xticks(bins, minor=True)
    plt.xticks([0.85, 0.9, 0.95, 1.0])
    plt.savefig("/home/max/Desktop/plots/roundness/Histogram.png")
    plt.show()

if False:
    ### HISTOGRAM for radii_q
    n, bins, patches = plt.hist(x=rq_rel_list, bins=list(np.linspace(0.86,1.0,int(0.15/0.01))),#bins='auto',
                                #color='#0504aa',
                                alpha=0.7,
                                rwidth=0.85,  #density=True,  # density macht nur % draus, durch *100
                                align="mid",
                                weights=rq_rel_list_weights  # y-Achse
                                )
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('relativer Radius')
    plt.ylabel('relative Häufigkeit')
    plt.title('Histogram der relativen Häufigkeit realtiver Radien (mit q-Multiplizität) für Verschiedene Auflösungen (GPD)', wrap=True)
    #plt.text(23, 45, r'$\mu=15, b=3$')
    plt.legend([str(x)+" GPD" for x in gpds])
    plt.ylim([0, 1])
    plt.xticks(bins, minor=True)
    plt.xticks([0.85,0.9,0.95,1.0])
    plt.savefig("/home/max/Desktop/plots/roundness/Histogram_q.png")
    plt.show()


if False:
    ### PLOT mean radius over GPD
    plt.figure()
    lines = plt.plot(gpds, r_rel_mean_list,
                     #gpds, r_rel_mean_PU_list,
                     gpds, rq_rel_mean_list,
                     #gpds, rq_rel_mean_PU_list,
                     gpds, r_rel_max_list,
                     gpds, r_rel_min_list)
    plt.setp(lines, ls="--", lw=1, marker=".")
    plt.legend([r"$\bar{r}$ Gitterpunte einfach gezählt",
                #"PU",
                r"$\bar{r}$ Gitterpunkte mit Anzahl der Verbindungen zu Fluidknoten",
                #"q PU",
                "$r_{max}$",
                "$r_{min}$"])
    plt.title("Mittlerer, maximaler und minimaler relativer Radius in Abhängigkeit des Durchmessers in Gitterpunkten (GPD) für DpY = "+str(domain_width_in_D), wrap=True)
    plt.grid(visible=True)
    plt.ylim([0.6,1.01])
    plt.xlim([0,101])
    plt.xlabel("GPD")
    plt.savefig("/home/max/Desktop/plots/roundness/mittlererRadius")
    plt.show()

pass