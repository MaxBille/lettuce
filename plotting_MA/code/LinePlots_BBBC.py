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
matplotlib.rcParams.update({'figure.figsize': (7.22433,6)})

### DATA I/O settings
data_base_path = "/home/mbille/Desktop/MA_Paraview_DataAndScreenshots-for-Thesis/FIG_BBBC"
output_base_path = "/home/mbille/lettuce/plotting_MA"
plot_batch_name = "test_LinePlotting_BBBC"

if not os.path.exists(output_base_path + "/" + plot_batch_name):
    os.makedirs(output_base_path + "/" + plot_batch_name)

cobc_variants = ["KBC_FWBB", "REG_ibb05"]
#bc_variants = ["FWBB", "ibbd05"]
timesteps = [30000, 30001, 30002, 30003]
yrel_variants = ["yrel0", "yrel4"]

# CSV format:
# "TimeStep", "p" ,"ux" ,"uy" ,"uz" ,"Points:0" ,"Points:1" ,"Points:2"
# 0            1    2     3     4     5   x        6   y        7  z

data_fig_path = "/temporal_osc_over_BBBC"
#datasets = np.zeros((len(co_variants), len(bc_variants), len(timesteps), len(yrel_variants)))
datasets = [[[[] for _ in range(len(yrel_variants))] for _ in range(len(timesteps))] for _ in range(len(cobc_variants))]
for cobc_index,cobc_variant in enumerate(cobc_variants):
    for timestep_index, timestep in enumerate(timesteps):
        for yrel_index,yrel_variant in enumerate(yrel_variants):
            datasets[cobc_index][timestep_index][yrel_index] = np.unique(np.genfromtxt(data_base_path + data_fig_path + "/" +cobc_variant+"_low_step"+str(timestep)+"_"+yrel_variant+ ".csv", delimiter=",", skip_header=1), axis=0)
            sort_indices = np.argsort(datasets[cobc_index][timestep_index][yrel_index][:, 5])
            datasets[cobc_index][timestep_index][yrel_index] = datasets[cobc_index][timestep_index][yrel_index][sort_indices]
datasets = np.array(datasets)
# 2,4,2,80,8
# cobc, timestep, yrel, x, OBS

data_reg_fwbb_yrel0 = np.unique(np.genfromtxt(data_base_path + data_fig_path + "/REG_FWBB_low_step30000_yrel0.csv", delimiter=",", skip_header=1), axis=0)
data_reg_fwbb_yrel4 = np.unique(np.genfromtxt(data_base_path + data_fig_path + "/REG_FWBB_low_step30000_yrel4.csv", delimiter=",", skip_header=1), axis=0)

sort_indices = np.argsort(data_reg_fwbb_yrel0[:, 5])
data_reg_fwbb_yrel0 = data_reg_fwbb_yrel0[sort_indices]
sort_indices = np.argsort(data_reg_fwbb_yrel4[:, 5])
data_reg_fwbb_yrel4 = data_reg_fwbb_yrel4[sort_indices]

### fig_bbbc_1: KBC

fig_bbbc_1, axs_bbbc = plt.subplots(3,2, sharex='col', sharey='row')
# 2 collision operators -> 2 FIGURES!
# p, ux, uy (3 observables)  -> 3 rows
# 2 positions... yrel -> 2 COLS
# 4 timesteps (within one plot)

# p @ yrel0
axs_bbbc[0,0].plot(datasets[0,0,0][:,5],datasets[0,0,0][:,1],
                   datasets[0,1,0][:,5],datasets[0,1,0][:,1],
                   datasets[0,2,0][:,5],datasets[0,2,0][:,1],
                   datasets[0,3,0][:,5],datasets[0,3,0][:,1]
                   )

# p @ yrel4
axs_bbbc[0,1].plot(datasets[0,0,1][:,5],datasets[0,0,1][:,1],
                   datasets[0,1,1][:,5],datasets[0,1,1][:,1],
                   datasets[0,2,1][:,5],datasets[0,2,1][:,1],
                   datasets[0,3,1][:,5],datasets[0,3,1][:,1]
                   )

# ux @yrel0
axs_bbbc[1,0].plot(datasets[0,0,0][:,5],datasets[0,0,0][:,2],
                   datasets[0,1,0][:,5],datasets[0,1,0][:,2],
                   datasets[0,2,0][:,5],datasets[0,2,0][:,2],
                   datasets[0,3,0][:,5],datasets[0,3,0][:,2]
                   )

# ux @yrel4
axs_bbbc[1,1].plot(datasets[0,0,1][:,5],datasets[0,0,1][:,2],
                   datasets[0,1,1][:,5],datasets[0,1,1][:,2],
                   datasets[0,2,1][:,5],datasets[0,2,1][:,2],
                   datasets[0,3,1][:,5],datasets[0,3,1][:,2]
                   )

# uy @yrel0
axs_bbbc[2,0].plot(datasets[0,0,0][:,5],datasets[0,0,0][:,3],
                   datasets[0,1,0][:,5],datasets[0,1,0][:,3],
                   datasets[0,2,0][:,5],datasets[0,2,0][:,3],
                   datasets[0,3,0][:,5],datasets[0,3,0][:,3]
                   )

# uy @urel4
axs_bbbc[2,1].plot(datasets[0,0,1][:,5],datasets[0,0,1][:,3],
                   datasets[0,1,1][:,5],datasets[0,1,1][:,3],
                   datasets[0,2,1][:,5],datasets[0,2,1][:,3],
                   datasets[0,3,1][:,5],datasets[0,3,1][:,3]
                   )

# x-axis labels
axs_bbbc[2,0].set_xlabel(r"$x_{LU}$")
axs_bbbc[2,1].set_xlabel(r"$x_{LU}$")

# y-axis labels
axs_bbbc[0,0].set_ylabel(r"$p_{LU}$")
axs_bbbc[1,0].set_ylabel(r"$u_{x,LU}$")
axs_bbbc[2,0].set_ylabel(r"$u_{y,LU}$")

# y-axis limits
axs_bbbc[0,0].set_ylim([-0.00025,0.00025])
axs_bbbc[1,0].set_ylim([-0.005,0.005])
axs_bbbc[2,0].set_ylim([-0.01,0.01])

# x-axis limits
axs_bbbc[2,0].set_xlim([0,80])
axs_bbbc[2,1].set_xlim([0,80])

axs_bbbc[0,0].set_title(r"at $y_{LU} = 1$")
axs_bbbc[0,1].set_title(r"at $y_{LU} = 5$")

axs_bbbc[2,1].legend(labels=["KBC fwbb i=30000", "KBC fwbb i=30001", "KBC fwbb i=30002", "KBC fwbb i=30003"],fontsize=6)
plt.suptitle(f"p/ux/uy(x,ti), KBC, FWBB, lowInlet, RES8, i30000+, res8")
plt.savefig(output_base_path + "/" + plot_batch_name + "/" + "test_lowInlet_BBBC_KBC_FWBB")
plt.close(fig_bbbc_1)

####################################

### fig_bbbc_2: KBC

fig_bbbc_2, axs_bbbc = plt.subplots(3,2, sharex='col', sharey='row')
# 2 collision operators -> 2 FIGURES!
# p, ux, uy (3 observables)  -> 3 rows
# 2 positions... yrel -> 2 COLS
# 4 timesteps (within one plot)

# p @ yrel0
axs_bbbc[0,0].plot(data_reg_fwbb_yrel0[:,5],data_reg_fwbb_yrel0[:,1],
                   datasets[1,0,0][:,5],datasets[1,0,0][:,1],
                   datasets[1,1,0][:,5],datasets[1,1,0][:,1],
                   datasets[1,2,0][:,5],datasets[1,2,0][:,1],
                   datasets[1,3,0][:,5],datasets[1,3,0][:,1]
                   )

# p @ yrel4
axs_bbbc[0,1].plot(data_reg_fwbb_yrel4[:,5],data_reg_fwbb_yrel4[:,1],
                   datasets[1,0,1][:,5],datasets[1,0,1][:,1],
                   datasets[1,1,1][:,5],datasets[1,1,1][:,1],
                   datasets[1,2,1][:,5],datasets[1,2,1][:,1],
                   datasets[1,3,1][:,5],datasets[1,3,1][:,1]
                   )

# ux @yrel0
axs_bbbc[1,0].plot(data_reg_fwbb_yrel0[:,5],data_reg_fwbb_yrel0[:,2],
                   datasets[1,0,0][:,5],datasets[1,0,0][:,2],
                   datasets[1,1,0][:,5],datasets[1,1,0][:,2],
                   datasets[1,2,0][:,5],datasets[1,2,0][:,2],
                   datasets[1,3,0][:,5],datasets[1,3,0][:,2]
                   )

# ux @yrel4
axs_bbbc[1,1].plot(data_reg_fwbb_yrel4[:,5],data_reg_fwbb_yrel4[:,2],
                   datasets[1,0,1][:,5],datasets[1,0,1][:,2],
                   datasets[1,1,1][:,5],datasets[1,1,1][:,2],
                   datasets[1,2,1][:,5],datasets[1,2,1][:,2],
                   datasets[1,3,1][:,5],datasets[1,3,1][:,2]
                   )

# uy @yrel0
axs_bbbc[2,0].plot(data_reg_fwbb_yrel0[:,5],data_reg_fwbb_yrel0[:,3],
                   datasets[1,0,0][:,5],datasets[1,0,0][:,3],
                   datasets[1,1,0][:,5],datasets[1,1,0][:,3],
                   datasets[1,2,0][:,5],datasets[1,2,0][:,3],
                   datasets[1,3,0][:,5],datasets[1,3,0][:,3]
                   )

# uy @urel4
axs_bbbc[2,1].plot(data_reg_fwbb_yrel4[:,5],data_reg_fwbb_yrel4[:,3],
                   datasets[1,0,1][:,5],datasets[1,0,1][:,3],
                   datasets[1,1,1][:,5],datasets[1,1,1][:,3],
                   datasets[1,2,1][:,5],datasets[1,2,1][:,3],
                   datasets[1,3,1][:,5],datasets[1,3,1][:,3]
                   )

# x-axis labels
axs_bbbc[2,0].set_xlabel(r"$x_{LU}$")
axs_bbbc[2,1].set_xlabel(r"$x_{LU}$")

# y-axis labels
axs_bbbc[0,0].set_ylabel(r"$p_{LU}$")
axs_bbbc[1,0].set_ylabel(r"$u_{x,LU}$")
axs_bbbc[2,0].set_ylabel(r"$u_{y,LU}$")

# y-axis limits
# axs_bbbc[0,0].set_ylim([-0.001,0.001])
# axs_bbbc[1,0].set_ylim([-0.02,0.02])
# axs_bbbc[2,0].set_ylim([-0.02,0.02])
axs_bbbc[0,0].set_ylim([-0.00025,0.00025])
axs_bbbc[1,0].set_ylim([-0.01,0.01])
axs_bbbc[2,0].set_ylim([-0.01,0.01])

# x-axis limits
axs_bbbc[2,0].set_xlim([0,80])
axs_bbbc[2,1].set_xlim([0,80])

axs_bbbc[0,0].set_title(r"at $y_{LU} = 1$")
axs_bbbc[0,1].set_title(r"at $y_{LU} = 5$")

axs_bbbc[2,1].legend(labels=["REG fwbb i30000","REG ibbd0.5 i=30000", "REG ibbd0.5 i=30001", "REG ibbd0.5 i=30002", "REG ibbd0.5 i=30003"],fontsize=6)
plt.suptitle(f"p/ux/uy(x,ti), REG, FWBB/ibbd0.5, lowInlet, RES8, i30000+, res8")
plt.savefig(output_base_path + "/" + plot_batch_name + "/" + "test_lowInlet_BBBC_REG")
plt.close(fig_bbbc_2)