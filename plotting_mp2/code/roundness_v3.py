### script to investigate the effect of discretisation on the shape of a circle
# - .png outputs can be toggled through True/False values at if-statements (see below)
#
# inputs: see ###INPUT1 and ###INPUT2 below
# - True/False values in if statements
# - 1. list of gpd values (use a few values for histogram, for example: [20,21,22], use many values for relative-radius-plot)
# - 2. domain_width_in_D and domain_length_in_D
#
# outputs:
# - plots mask of circle
# - plots histogram of relative radii (distance between center and nodes and the circumference of the circle
#   a) each grid point (node) counted once
#   b) each grid point (node) counted according to the number of links to fluid nodes
# - plot for mean, max, min relative radius for all GPD values

import lettuce as lt
import matplotlib.pyplot as plt
import numpy as np
import torch
import datetime
import os
import shutil

###INPUT1:
timestamp = datetime.datetime.now()
timestamp = timestamp.strftime("%y%m%d")+"_"+timestamp.strftime("%H%M%S")
name = "roundness_criteria_engl"
dir_name = "../roundness_criteria/data_" + str(timestamp) + "_" + name
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)
if not os.path.isdir(dir_name+"/masks"):
    os.mkdir(dir_name + "/masks")

output_file = open(dir_name + "/console_output.txt", "a")

show = False

# List of Diameters (in GPD) to measure:
gpds = [9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,30,35,40,45,50,60,70,71,72,73,74,75,80,90,100] #(np.arange(150)+5)
gpds = [2,5,10,11,12,20,21,22,52,53, 60,70,80,90,100, 150]
#gpds = [20,21,22]
#gpds = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,30,35,40,45,50,60,70,71,72,73,74,75,80,90,100,150,151,152,153,154,155,160]
gpds = np.arange(0,60)+100
gpds = [20,21,22]
#gpds = np.arange(5,151)

# lists for plotting
r_rel_list = []
r_rel_list_weights = []  # weights needed for histogram
rq_rel_list = []
rq_rel_list_weights = []

r_rel_mean_list = []
rq_rel_mean_list = []

r_rel_mean_PU_list = []
rq_rel_mean_PU_list = []

r_rel_max_list = []
r_rel_min_list = []

area_rel_list = []

# calculate radii etc. for all GPDs
for i in gpds:
    output_file.write("#######")
    print("#######")
    gridpoints_per_diameter = i  # gp_per_D -> this defines the resolution ( D_LU = GPD+1)
    ### INPUT2
    domain_width_in_D = 1+(2/gridpoints_per_diameter)  # D/Y 19  -> this defines the domain-size and total number of Lattice-Nodes
    domain_length_in_D = 1+(2/gridpoints_per_diameter) #2*domain_width_in_D 2*domain_width_in_D # D/X

        # ### GPD-correction doesn't make sense here, but keep in mind the combination of GPD and DpY is not completely arbitrary
        # # if DpY is even, resulting GPD can't be odd for symmetrical cylinder and channel
        # # ...if DpY is even, GPD will be corrected to even GPD for symmetrical cylinder
        # # ...use odd DpY to use odd GPD
        # gpd_correction=False
        # if False:#domain_width_in_D % 2 == 0 and gridpoints_per_diameter % 2 != 0:
        #     gpd_correction = True   # gpd_was_corrected-flag
        #     gpd_setup = gridpoints_per_diameter   # store old gpd for output
        #     gridpoints_per_diameter = int(gridpoints_per_diameter/2)*2   # make gpd even
        #     print("(!) domain_width_in_D is even, gridpoints_per_diameter will be "+str(gridpoints_per_diameter)+". Use odd domain_width_in_D to enable use of odd GPD!")

    output_file.write("\nshape_LU:" + str(round(gridpoints_per_diameter*domain_length_in_D)) + " x " + str(round(gridpoints_per_diameter*domain_width_in_D)))
    print("shape_LU:", round(gridpoints_per_diameter*domain_length_in_D), "x", round(gridpoints_per_diameter*domain_width_in_D))

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
                        rand_xq.append(a[p])
                        rand_yq.append(b[p])
                except IndexError:
                    pass  # just ignore this iteration since there is no neighbor there

    rand_x, rand_y = np.where(rand_mask)  # Liste aller Rand-x- und y-Koordinaten
    x_pos = sum(rand_x)/len(rand_x)  # x_Koordinate des Kreis-Zentrums
    y_pos = sum(rand_y)/len(rand_y)  # y-Koordinate des Kreis-Zentrums

        ###DOESN'T WORK: radii_q = np.sqrt(np.power(np.array(rand_xq)-x_pos, 2) + np.power(np.array(rand_yq)-y_pos, 2))  # Liste der Radien in LU (multiplizität, falls ein Randpunkt mehrere Links zu Fluidknoten hat

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
    radii_relative = radii / (radius_LU-0.5)  # (substract 0.5 because "true" boundary location is 0.5LU further out than node-coordinates)
    radii_q_relative = radii_q / (radius_LU-0.5)

    # append to GLOBAL lists for plotting
    r_rel_list.append(radii_relative)
    r_rel_list_weights.append(np.ones_like(radii_relative) / len(radii_relative))  # needed for histogram
    rq_rel_list.append(radii_q_relative)
    rq_rel_list_weights.append(np.ones_like(radii_q_relative) / len(radii_q_relative))

    # calc. mean rel_radius
    r_rel_mean = sum(radii_relative)/len(radii_relative)
    rq_rel_mean = sum(radii_q_relative)/len(radii_q_relative)

    r_rel_mean_list.append(r_rel_mean)
    rq_rel_mean_list.append(rq_rel_mean)

        # ### all relative radii in relation to D/2 in PU (D=GPD+1 (!))
        # radii_relative_PU = (radii + 0.5) / (radius_LU + 0.5)
        # radii_q_relative_PU = (radii_q + 0.5) / (radius_LU + 0.5)
        #
        # # mean radius in PU
        # r_rel_mean_PU = sum(radii_relative_PU) / len(radii_relative_PU)
        # rq_rel_mean_PU = sum(radii_q_relative_PU) / len(radii_q_relative_PU)
        #
        # r_rel_mean_PU_list.append(r_rel_mean_PU)
        # rq_rel_mean_PU_list.append(rq_rel_mean_PU)

    # append max/min radii
    r_rel_max_list.append(r_max/(radius_LU-0.5))
    r_rel_min_list.append(r_min/(radius_LU-0.5))

    ### AREA calculation
    area_theory = np.pi*(gridpoints_per_diameter/2)**2  # area = pi*r² in LU²
    area = len(a)  # area in LU = number of nodes, because every node has a cell of 1LU x 1LU around it

    area_rel_list.append(area/area_theory)

    from collections import Counter
    print("GPD: ", gridpoints_per_diameter)
    print("radii: ", Counter(radii))
    print("radii_q: ", Counter(radii_q))
    output_file.write("\nGPD: " + str(gridpoints_per_diameter))
    output_file.write("\nradii: " + str(Counter(radii)))
    output_file.write("\nradii_q: " + str(Counter(radii_q))+"\n\n")

    if True:  # toggle mask output to .png
        ### PLOT rand_mask
        plt.figure()
        plt.imshow(rand_mask)
        #plt.xticks(np.arange(gridpoints_per_diameter + 2), minor=True)
        #plt.yticks(np.arange(gridpoints_per_diameter + 2), minor=True)
        ax = plt.gca()
        xmin, xmax = ax.get_xlim()
        ymax, ymin = ax.get_ylim()
        if gridpoints_per_diameter >= 10:
            plt.xticks(np.arange(0, xmax, int(xmax/10)))
            plt.yticks(np.arange(0, ymax, int(ymax/10)))
        else:
            plt.xticks(np.arange(0, xmax, 1))
            plt.yticks(np.arange(0, ymax, 1))
        plt.title("GPD = "+str(gridpoints_per_diameter))
        ax.set_xticks(np.arange(-.5, xmax, 1), minor=True)
        ax.set_yticks(np.arange(-.5, ymax, 1), minor=True)
        if gridpoints_per_diameter < 30:
            ax.grid(which="minor", color="k", axis='both', linestyle='-', linewidth=2)
        elif gridpoints_per_diameter < 70:
            ax.grid(which="minor", color="k", axis='both', linestyle='-', linewidth=1)
        elif gridpoints_per_diameter < 100:
            ax.grid(which="minor", color="k", axis='both', linestyle='-', linewidth=0.5)
        elif gridpoints_per_diameter < 150:
            ax.grid(which="minor", color="k", axis='both', linestyle='-', linewidth=0.25)
        plt.savefig(dir_name + "/masks/randMask_GPD" + str(gridpoints_per_diameter) + ".png")
        if show:
            plt.show()
        plt.close()

if len(gpds) <= 10:  # toggle HISTOGRAMM of radii
    ### HISTOGRAM for radii
    n, bins, patches = plt.hist(x=r_rel_list,  #bins=list(np.linspace(0.86,1.0,int(0.15/0.01))),
                                #bins='auto',
                                #color='#0504aa',
                                alpha=0.7,
                                rwidth=0.85,  #density=True,  # density macht nur % draus, durch *100
                                align="mid",
                                weights=r_rel_list_weights  # y-Achse
                                )
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('relative radius')  #plt.xlabel('relativer Radius')
    plt.ylabel('relative frequency')  #plt.ylabel('relative Häufigkeit')
    # plt.title(
    #     'Histogram der relativen Häufigkeit realtiver Radien für verschiedene Auflösungen (GPD)',
    #     wrap=True)
    # plt.text(23, 45, r'$\mu=15, b=3$')
    plt.legend([str(x) + " GPD" for x in gpds])
    plt.ylim([0, 1])
    #plt.xticks(bins, minor=True)
    #plt.xticks([0.85, 0.9, 0.95, 1.0])
    plt.savefig(dir_name+"/Histogram.png")
    if show:
        plt.show()
    else:
        plt.close()
else:
    print("too many gpd for histogram")
    output_file.write("\nWARNING: too many GPD for histogram!\n")

if len(gpds) <= 10:  # toggle HISTOGRAMM of radii with q-multiplicity (links taken into account)
    ### HISTOGRAM for radii_q
    n, bins, patches = plt.hist(x=rq_rel_list, #bins=list(np.linspace(0.86,1.0,int(0.15/0.01))),
                                #bins='auto',
                                #color='#0504aa',
                                alpha=0.7,
                                rwidth=0.85,  #density=True,  # density macht nur % draus, durch *100
                                align="mid",
                                weights=rq_rel_list_weights  # y-Achse
                                )
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('relative radius')  # plt.xlabel('relativer Radius')
    plt.ylabel('relative frequency')  # plt.ylabel('relative Häufigkeit')
    # plt.title('Histogram der relativen Häufigkeit realtiver Radien (mit q-Multiplizität) für Verschiedene Auflösungen (GPD)', wrap=True)
    #plt.text(23, 45, r'$\mu=15, b=3$')
    plt.legend([str(x)+" GPD" for x in gpds])
    plt.ylim([0, 1])
    #plt.xticks(bins, minor=True)
    #plt.xticks([0.85, 0.9, 0.95, 1.0])
    plt.savefig(dir_name + "/Histogram_q.png")
    if show:
        plt.show()
    else:
        plt.close()
else:
    print("too many gpd for histogram")
    output_file.write("\nWARNING: too many GPD for histogram_q!\n")



if True:  # toggle plot for mean, max, min radius over all GPD
    ### PLOT mean radius over GPD
    plt.figure()
    lines = plt.plot(gpds, r_rel_mean_list,
                     #gpds, r_rel_mean_PU_list,
                     gpds, rq_rel_mean_list,
                     #gpds, rq_rel_mean_PU_list,
                     gpds, r_rel_max_list,
                     gpds, r_rel_min_list)
    plt.setp(lines, ls="--", lw=1, marker=".")
    # plt.legend([r"$\bar{r}$ Gitterpunte einfach gezählt",
    #             #"PU",
    #             r"$\bar{r}$ Gitterpunkte mit Anzahl der Verbindungen zu Fluidknoten",
    #             #"q PU",
    #             "$r_{max}$",
    #             "$r_{min}$"])
    plt.legend([r"$\bar{r}$ lattice sites neighboring fluid",
                # "PU",
                r"$\bar{r}$ lattice links pointing to fluid",
                # "q PU",
                "$r_{max}$",
                "$r_{min}$"])
    #plt.title("Mittlerer, maximaler und minimaler relativer Radius in Abhängigkeit des Durchmessers in Gitterpunkten (GPD)", wrap=True)
    plt.grid(visible=True)
    plt.ylim([0.6, 1.01*max(r_rel_max_list)])
    plt.xlim([0, max(gpds)+1])
    plt.xticks(np.linspace(0, int(max(gpds)/10)*10, int((max(gpds))/10)+1))
    plt.xlabel("GPD")
    plt.savefig(dir_name + "/mittlererRadius.png")
    if show:
        plt.show()
    else:
        plt.close()

if True:  # toggle plot for relative area over all GPD
    plt.figure()
    lines = plt.plot(gpds, area_rel_list)
    plt.setp(lines, ls="--", lw=1, marker=".")
    # plt.legend([r"relative Flaeche im Verhaeltnis zur theoretischen Flaeche r²\pi",
    #             #"PU",
    #             r"$\bar{r}$ Gitterpunkte mit Anzahl der Verbindungen zu Fluidknoten",
    #             #"q PU",
    #             "$r_{max}$",
    #             "$r_{min}$"])
    #plt.title(r"relative praktische Kreisflaeche (Knotenzahl/($\pi$(gpd/2)²), in Abhängigkeit des Durchmessers in Gitterpunkten (GPD)", wrap=True)
    plt.grid(visible=True)
    #plt.ylim([0.6, 1.01])
    plt.xlim([0, max(gpds) + 1])
    plt.xticks(np.linspace(0, int(max(gpds) / 10) * 10, int((max(gpds)) / 10) + 1))
    plt.xlabel("GPD")
    plt.ylabel("relative area")  #plt.ylabel("relative Flaeche")
    plt.savefig(dir_name + "/relativeFleache.png")
    if show:
        plt.show()
    else:
        plt.close()

output_file.close()
pass