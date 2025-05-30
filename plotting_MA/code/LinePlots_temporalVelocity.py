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
matplotlib.rcParams.update({'font.size': 8}) # font size was 11

### DATA I/O settings
data_base_path = "/home/mbille/Desktop/MA_Paraview_DataAndScreenshots-for-Thesis/FIG_Einlassgeschwindigkeit_Zeitpunkt"
output_base_path = "/home/mbille/lettuce/plotting_MA"
plot_batch_name = "test_LinePlotting_SmoothTemporalVelocity"

timesteps = [2500, 5000, 7500, 10000] #[51, 2500, 5000, 7500, 10000]
inlet_velocities = [0.00432327, 0.00864827, 0.01297327, 0.01727728733] #[0.0000865, 0.00432327, 0.00864827, 0.01297327, 0.01727728733]
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
dataset_names = ['oben/unten periodisch', 'oben/unten periodisch', 'oben/unten HWBB', 'oben/unten HWBB'] # ['noBBBC_yrel0',
                 # 'noBBBC_yrel5',
                 # 'IBBd05_yrel0',
                 # 'IBBd05_yrel5']

# Mapping: [noBBBC_yrel0, IBBd05_yrel0, noBBBC_yrel5, IBBd05_yrel5]
# Ziel: 4x Figures mit je 2x2 Layout

plot_configs = [
    {"title": "yrel0",              "norm": False, "dataset_idxs": [0, 2], "filename": "smoothTemporalProfiles_noBBBCvsHWBB_yrel0", "ylimsp": [-0.00015, 0.00015], "ylimsu": [-0.005, 0.02], "legloc": "upper right"},
    {"title": "yrel0 (normalized)", "norm": True, "dataset_idxs": [0, 2], "filename": "smoothTemporalProfiles_noBBBCvsHWBB_yrel0_normalized", "ylimsp": [-0.01, 0.01], "ylimsu": [-0.2, 1], "legloc": "upper right"},
    {"title": "yrel5",              "norm": False, "dataset_idxs": [1, 3], "filename": "smoothTemporalProfiles_noBBBCvsHWBB_yrel5", "ylimsp": [-0.00008, 0.00008], "ylimsu": [-0.002, 0.002], "legloc": "lower right"},
    {"title": "yrel5 (normalized)", "norm": True, "dataset_idxs": [1, 3], "filename": "smoothTemporalProfiles_noBBBCvsHWBB_yrel5_normalized", "ylimsp": [-0.008, 0.008], "ylimsu": [-0.2, 0.2], "legloc": "lower right"},
]

for config in plot_configs:
    fig, axs = plt.subplots(2, 2, sharex='col',sharey='row', figsize=(7, 4), constrained_layout=True)
    axs = axs.reshape(2, 2)  # [row][col]

    for col, dataset_index in enumerate(config["dataset_idxs"]):
        data = datasets[dataset_index]
        name = dataset_names[dataset_index]

        for timestep_index in range(len(timesteps)):
            x_vals = data[timestep_index, :, 5]
            p_vals = data[timestep_index, :, 1]
            u_vals = data[timestep_index, :, 2]

            if config["norm"]:
                norm_factor = inlet_velocities[timestep_index]
                p_vals = p_vals / norm_factor
                u_vals = u_vals / norm_factor

            axs[0][col].plot(x_vals, p_vals,
                             label=fr"$t_{{LU}}= \overline{{{timesteps[timestep_index]}\pm 50}}$")
            axs[1][col].plot(x_vals, u_vals,
                             label=fr"$t_{{LU}}= \overline{{{timesteps[timestep_index]}\pm 50}}$")
        #fr"p(x,y = 0,z = 5, i = $\overline{{{timesteps[timestep_index]}\pm 50}}$)"
        axs[0][col].set_title(f"{name}")
        axs[0][col].set_xlim([0, 80])
        axs[0][0].set_ylabel(r"$p$" if not config["norm"] else r"$p/u_{in}$")
        axs[1][col].set_xlabel(r"$x_{LU}$")
        axs[1][0].set_ylabel(r"$u_x$" if not config["norm"] else r"$u_x/u_{in}$")
        axs[1][col].set_xlim([0, 80])

        axs[0][col].set_ylim(config["ylimsp"])
        axs[1][col].set_ylim(config["ylimsu"])

        # if col == 1:
        #     axs[0][col].legend(fontsize=5)
        #     axs[1][col].legend(fontsize=5)

    axs[1][1].legend(loc=config["legloc"])
    #fig.suptitle(f"Pressure and Velocity (uₓ) Profiles – {config['title']}", fontsize=10)
    savepath = f"{output_base_path}/{plot_batch_name}/{config['filename']}.png"
    plt.savefig(savepath, dpi=300)
    plt.close(fig)