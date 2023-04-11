import lettuce as lt
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

import warnings
import numpy as np
from lettuce.unit import UnitConversion
from lettuce.util import append_axes
from lettuce.boundary import EquilibriumBoundaryPU, BounceBackBoundary, HalfwayBounceBackBoundary, FullwayBounceBackBoundary, EquilibriumOutletP, AntiBounceBackOutlet
from lettuce.flows.obstaclemax import ObstacleMax

import torch
import time
import datetime
import os
import shutil

for i in (np.arange(45)+5):
    ### Simulationsparameter - Steuerung
    re = 200   # Reynoldszahl
    Ma = 0.05     # Machzahl
    n_steps = 126000    # Schrittzahl
    setup_diameter = 1  # D_PU = char_length_pu -> this defines the PU-Reference
    flow_velocity = 1  # U_PU = char_velocity_pu -> this defines the PU-Reference velocity (u_max of inflow)

    periodic_start = 0.9  # relative start of peak_finding for Cd_mean Measurement to cut of any transients

    gridpoints_per_diameter = i  # gp_per_D -> this defines the resolution ( D_LU = GPD+1)
    domain_width_in_D = 1  # D/Y  -> this defines the domain-size and total number of Lattice-Nodes
    domain_length_in_D = 1#2*domain_width_in_D  # D/X

    # if DpY is even, resulting GPD can't be odd for symmetrical cylinder and channel
    # ...if DpY is even, GPD will be corrected to even GPD for symemtrical cylinder
    # ...use odd DpY to use odd GPD
    gpd_correction=False
    if domain_width_in_D % 2 == 0 and gridpoints_per_diameter % 2 != 0:
        gpd_correction = True   # gpd_was_corrected-flag
        gpd_setup = gridpoints_per_diameter   # store old gpd for output
        gridpoints_per_diameter = int(gridpoints_per_diameter/2)*2   # make gpd even
        print("(!) domain_width_in_D is even, gridpoints_per_diameter will be "+str(gridpoints_per_diameter)+". Use odd domain_width_in_D to enable use of odd GPD!")

    T_target=140
    print("shape_LU:", gridpoints_per_diameter*domain_length_in_D, "x", gridpoints_per_diameter*domain_width_in_D)
    print("T with", n_steps, "steps:", round(n_steps * (setup_diameter/(gridpoints_per_diameter+1))*(Ma*1/np.sqrt(3)/flow_velocity),2), "seconds")
    print("n_steps to simulate 1 second:", round(((gridpoints_per_diameter+1)/setup_diameter)*(flow_velocity/(Ma*1/np.sqrt(3))),2), "steps")
    print("n_steps to simulate",T_target,"seconds:",T_target*round(((gridpoints_per_diameter+1)/setup_diameter)*(flow_velocity/(Ma*1/np.sqrt(3))),2), "steps")

    u_init = 0    # initiales Geschwindigkeitsfeld: # 0: uniform u=0, # 1: uniform u=1, # 2: parabolic, amplitude u_char_lu (similar to poiseuille-flow)
    perturb_init = True   # leichte Asymmetrie in Anfangs-Geschwindigkeitsverteilung -> triggert Karman'sche Wirbelstraße für Re>47
    bb_wall = False    # Randbedingungen der lateralen Kanalwände: True= bounce-back-walls und parabelförmiges Geschwindigkeisprodil am Einlass, False= periodic BC und uniforme Geschwindigkeit am Einlass
    cylinder = True    # Objekt: True = cylinder, False = no obstascle
    halfway = True    # BounceBackBoundary-Algorithmus: True=Halfway, False=Fullway
    drag_out = True    # drag_coefficient als observable-reporter
    lift_out = True    # lift_coefficient als observable-reporter
    vtk_fps=10    # FramesPerSecond (/PU) für vtk-output
    vtk_out=False   # is overwritten by output_save=False (see below)

    #exmpl: Re1000,steps25000,ny1000 braucht 43min
    #Bonn: Re200, steps200000,gpd20?, 400x200 braucht 10min
    #HBRS: Re200, steps100000, gpd20, 800x400 braucht 25min

    mlups_2060super = 20
    mlups_2080ti = 30

    if vtk_out:
        print("generates approx.", int(vtk_fps*(n_steps * (setup_diameter/(gridpoints_per_diameter+1))*(Ma*1/np.sqrt(3)/flow_velocity)))+1, ".vti/.vtk-frames")

    gridpoints = gridpoints_per_diameter**2*domain_length_in_D*domain_width_in_D
    print("No. of gridpoints:", gridpoints)
    print("estimated min. runtime on 2060super:", round(n_steps*gridpoints/(1e6*mlups_2060super),2), "seconds (", round(n_steps*gridpoints/(1e6*mlups_2060super)/60,2),"minutes )")
    print("estimated min. runtime on 2080ti:   ", round(n_steps*gridpoints/(1e6*mlups_2080ti),2), "seconds (", round(n_steps*gridpoints/(1e6*mlups_2080ti)/60,2),"minutes )")

    ### Simulationssetup

    # lattice
    lattice = lt.Lattice(lt.D2Q9, "cuda:0", dtype=torch.float64)
    # stencil, device, dtype

    flow = ObstacleMax(reynolds_number=re, mach_number=Ma,
                       lattice=lattice,
                       char_length_pu=setup_diameter,
                       char_length_lu=gridpoints_per_diameter + 1,
                       char_velocity_pu=flow_velocity,
                       y_lu=domain_width_in_D * gridpoints_per_diameter,
                       x_lu=domain_length_in_D * gridpoints_per_diameter,
                       lateral_walls=bb_wall,
                       hwbb=halfway,
                       perturb_init=perturb_init,
                       u_init=u_init
                       )

    # define a Cylinder-Obstacle
    radius_LU = 0.5 * gridpoints_per_diameter
    y_pos_LU = 0.5 * gridpoints_per_diameter * domain_width_in_D + 0.5
    x_pos_LU = y_pos_LU

    xyz = tuple(np.linspace(1, n, n) for n in flow.shape)  # Tupel aus Listen inizes (1-n (nicht 0-based!))
    xLU, yLU = np.meshgrid(*xyz, indexing='ij')  # meshgrid aus den x-, y- (und z-)Indizes

    condition = np.sqrt((xLU - x_pos_LU) ** 2 + (yLU - y_pos_LU) ** 2) < radius_LU
    flow.obstacle_mask[np.where(condition)] = 1

    ### Simulations-Objekt (Simulator)
    tau = flow.units.relaxation_parameter_lu
    sim = lt.Simulation(flow, lattice,
                        lt.BGKCollision(lattice, tau),
                        # lt.RegularizedCollision(lattice, tau),
                        # lt.KBCCollision2D(lattice,tau),
                        lt.StandardStreaming(lattice)
                        )
    # Flow, Lattice-Parameter, KollisionsOperator-Objekt(Parameter), Streaming-Objekt

    MASK = lattice.convert_to_numpy(sim._boundaries[2].mask)

    plt.imshow(MASK)
    plt.savefig("/home/max/Desktop/GPD_shapes/new_GPD"+str(gridpoints_per_diameter))
pass