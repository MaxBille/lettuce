### IMPORT

# lettuce
import lettuce as lt
# unit conversion...

# os and data management
import datetime
import os, shutil
# from pyevtk.hl import imageToVTK
from glob import glob

# calculation
import numpy as np
# torch
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# plotting
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.rcParams['agg.path.chunksize'] = 1000

matplotlib.style.use('../figure_style.mplstyle')
#matplotlib.style.use('../figure_style_2column_singleplot.mplstyle')
matplotlib.rcParams.update({'lines.linewidth': 0.8})
#matplotlib.rcParams.update({'lines.linestyle': '--'})
#matplotlib.rcParams.update({'font.size': 8}) # font size was 11

### DATA I/O settings
data_base_path = "/home/mbille/Desktop/MA_Paraview_DataAndScreenshots-for-Thesis/FIG_Einlassgeschwindigkeit_Zeitpunkt"
output_base_path = "/home/mbille/lettuce/plotting_MA"
plot_batch_name = "test_LinePlotting"

if not os.path.exists(output_base_path + "/" + plot_batch_name):
    os.makedirs(output_base_path + "/" + plot_batch_name)

timesteps = [51, 2500, 5000, 7500, 10000]
inlet_velocities = [0.0000865, 0.00432327, 0.00864827, 0.01297327, 0.01727728733]
bc_variants = ["noBBBC", "IBBd0.5"]
yrel_variants = ["rel0", "yrel5"]
# CSV format:
# "TimeStep", "p" ,"ux" ,"uy" ,"uz" ,"Points:0" ,"Points:1" ,"Points:2"
# 0            1    2     3     4     5   x        6   y        7  z

data_noBBBC_yrel0 = []
data_noBBBC_yrel5 = []
data_IBBd05_yrel0 = []
data_IBBd05_yrel5 = []
for timestep_index in range(len(timesteps)):
    data_noBBBC_yrel0.append(np.genfromtxt(data_base_path + "/noBBBC_HL80_LinePlotData_step"+str(timesteps[timestep_index])+"_yrel0.csv", delimiter=",", skip_header=1))
    data_noBBBC_yrel5.append(np.genfromtxt(data_base_path + "/noBBBC_HL80_LinePlotData_step" + str(timesteps[timestep_index]) + "_yrel5.csv", delimiter=",", skip_header=1))
    data_IBBd05_yrel0.append(np.genfromtxt(data_base_path + "/IBBd0.5_HL80_LinePlotData_step" + str(timesteps[timestep_index]) + "_yrel0.csv", delimiter=",", skip_header=1))
    data_IBBd05_yrel5.append(np.genfromtxt(data_base_path + "/IBBd0.5_HL80_LinePlotData_step" + str(timesteps[timestep_index]) + "_yrel5.csv", delimiter=",", skip_header=1))

# convert to np-array and exclude all duplicate points from f'in ParaView
data_noBBBC_yrel0 = np.unique(np.array(data_noBBBC_yrel0),axis=1)
data_noBBBC_yrel5 = np.unique(np.array(data_noBBBC_yrel5),axis=1)
data_IBBd05_yrel0 = np.unique(np.array(data_IBBd05_yrel0),axis=1)
data_IBBd05_yrel5 = np.unique(np.array(data_IBBd05_yrel5),axis=1)
# ARRAY CONFIG: (timestep (5), x-coord (80), data-columns (8))

# SORT along x-axis
sort_indices = np.argsort(data_noBBBC_yrel0[:, :, 5], axis=1)
data_noBBBC_yrel0 = np.take_along_axis(data_noBBBC_yrel0, np.expand_dims(sort_indices, axis=2), axis=1)
sort_indices = np.argsort(data_noBBBC_yrel5[:, :, 5], axis=1)
data_noBBBC_yrel5 = np.take_along_axis(data_noBBBC_yrel5, np.expand_dims(sort_indices, axis=2), axis=1)
sort_indices = np.argsort(data_IBBd05_yrel0[:, :, 5], axis=1)
data_IBBd05_yrel0 = np.take_along_axis(data_IBBd05_yrel0, np.expand_dims(sort_indices, axis=2), axis=1)
sort_indices = np.argsort(data_IBBd05_yrel5[:, :, 5], axis=1)
data_IBBd05_yrel5 = np.take_along_axis(data_IBBd05_yrel5, np.expand_dims(sort_indices, axis=2), axis=1)

# list datasets:
datasets = [data_noBBBC_yrel0,
            data_noBBBC_yrel5,
            data_IBBd05_yrel0,
            data_IBBd05_yrel5]
dataset_names = ['noBBBC_yrel0',
                 'noBBBC_yrel5',
                 'IBBd05_yrel0',
                 'IBBd05_yrel5']


for dataset_index in range(len(datasets)):



    ### NOT NORMALIZED >>>


    ## PRESSURE
    p_fig, p_axs = plt.subplots()

    for timestep_index in range(len(timesteps)):
        p_axs.plot(datasets[dataset_index][timestep_index, :, 5],
                   datasets[dataset_index][timestep_index, :, 1],
                   #marker="", #linewidth=0.4,
                   label=fr"p(x,y = 0,z = 5, i = $\overline{{{timesteps[timestep_index]}\pm 50}}$)")
    p_axs.set_xlabel(r"$x_{LU}$")
    p_axs.set_ylabel(r"$p_{LU}$")

    # p_axs.set_xlim(left=100)
    # p_axs.set_xlim([9000,11000])

    p_axs.legend(fontsize=5)
    plt.suptitle(f"p(x,t), {dataset_names[dataset_index]}_pressure")
    plt.savefig(output_base_path + "/" + plot_batch_name + "/" + dataset_names[dataset_index] + "_pressure")
    plt.close(p_fig)


    ## VELOCITY x
    u_fig, u_axs = plt.subplots()

    for timestep_index in range(len(timesteps)):
        u_axs.plot(datasets[dataset_index][timestep_index, :, 5],
                   datasets[dataset_index][timestep_index, :, 2],
                   #marker="", #linewidth=0.4,
                   label=fr"$u_{{x}}$(x,y = 0,z = 5, i = $\overline{{{timesteps[timestep_index]}\pm 50}}$)")
    u_axs.set_xlabel(r"$x_{LU}$")
    u_axs.set_ylabel(r"$u_{LU}$")

    # u_axs.set_xlim(left=100)
    # u_axs.set_xlim([9000,11000])

    u_axs.legend(fontsize=5)
    plt.suptitle(f"u_x(x,t), {dataset_names[dataset_index]}_pressure")
    plt.savefig(output_base_path + "/" + plot_batch_name + "/" + dataset_names[dataset_index] + "_velocity_x")
    plt.close(u_fig)


    ## VELOCITY MAGNITUDE
    u_fig, u_axs = plt.subplots()

    for timestep_index in range(len(timesteps)):
        u_axs.plot(datasets[dataset_index][timestep_index, :, 5],
                   np.sqrt(datasets[dataset_index][timestep_index, :, 2]**2+datasets[dataset_index][timestep_index, :, 3]**2+datasets[dataset_index][timestep_index, :, 4]**2),
                   #marker="", #linewidth=0.4,
                   label=fr"$u_{{mag}}$(x,y = 0,z = 5, i = $\overline{{{timesteps[timestep_index]}\pm 50}}$)")
    u_axs.set_xlabel(r"$x_{LU}$")
    u_axs.set_ylabel(r"$u_{LU}$")

    # u_axs.set_xlim(left=100)
    # u_axs.set_xlim([1,80])
    # u_axs.set_ylim(top=0.005)

    u_axs.legend(fontsize=5)
    plt.suptitle(f"u_mag(x,t), {dataset_names[dataset_index]}_velocity_magnitude")
    plt.savefig(output_base_path + "/" + plot_batch_name + "/" + dataset_names[dataset_index] + "_velocity_magnitude")
    plt.close(u_fig)


    ## NORMALIZED >>>

    ## PRESSURE
    p_fig, p_axs = plt.subplots()

    for timestep_index in range(len(timesteps)):
        p_axs.plot(datasets[dataset_index][timestep_index, :, 5],
                   datasets[dataset_index][timestep_index, :, 1]/inlet_velocities[timestep_index],
                   #marker="", #linewidth=0.4,
                   label=fr"p(x,y = 0,z = 5, i = $\overline{{{timesteps[timestep_index]}\pm 50}}$)")
    p_axs.set_xlabel(r"$x_{LU}$")
    p_axs.set_ylabel(r"$p_{LU}$")

    # p_axs.set_xlim(left=100)
    # p_axs.set_xlim([9000,11000])

    p_axs.legend(fontsize=5)
    plt.suptitle(f"p(x,t), {dataset_names[dataset_index]}_pressure_normalized")
    plt.savefig(output_base_path + "/" + plot_batch_name + "/" + dataset_names[dataset_index] + "_pressure_normalized")
    plt.close(p_fig)


    ## VELOCITY x
    u_fig, u_axs = plt.subplots()

    for timestep_index in range(len(timesteps)):
        u_axs.plot(datasets[dataset_index][timestep_index, :, 5],
                   datasets[dataset_index][timestep_index, :, 2]/inlet_velocities[timestep_index],
                   #marker="", #linewidth=0.4,
                   label=fr"$u_{{x}}$(x,y = 0,z = 5, i = $\overline{{{timesteps[timestep_index]}\pm 50}}$)")
    u_axs.set_xlabel(r"$x_{LU}$")
    u_axs.set_ylabel(r"$u_{LU}$")

    # u_axs.set_xlim(left=100)
    # u_axs.set_xlim([9000,11000])

    u_axs.legend(fontsize=5)
    plt.suptitle(f"u_x(x,t), {dataset_names[dataset_index]}_pressure_normalized")
    plt.savefig(output_base_path + "/" + plot_batch_name + "/" + dataset_names[dataset_index] + "_velocity_x_normalized")
    plt.close(u_fig)


    ## VELOCITY MAGNITUDE
    u_fig, u_axs = plt.subplots()

    for timestep_index in range(len(timesteps)):
        u_axs.plot(datasets[dataset_index][timestep_index, :, 5],
                   np.sqrt(datasets[dataset_index][timestep_index, :, 2]**2+datasets[dataset_index][timestep_index, :, 3]**2+datasets[dataset_index][timestep_index, :, 4]**2)/inlet_velocities[timestep_index],
                   #marker="", #linewidth=0.4,
                   label=fr"$u_{{mag}}$(x,y = 0,z = 5, i = $\overline{{{timesteps[timestep_index]}\pm 50}}$)")
    u_axs.set_xlabel(r"$x_{LU}$")
    u_axs.set_ylabel(r"$u_{LU}$")

    # u_axs.set_xlim(left=100)
    # u_axs.set_xlim([1,80])
    # u_axs.set_ylim(top=0.005)

    u_axs.legend(fontsize=5)
    plt.suptitle(f"u_mag(x,t),{dataset_names[dataset_index]}_velocity_magnitude_normalized")
    plt.savefig(output_base_path + "/" + plot_batch_name + "/" + dataset_names[dataset_index] + "_velocity_magnitude_normalized")
    plt.close(u_fig)

