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
data_base_path = "/home/mbille/Desktop/MA_Paraview_DataAndScreenshots-for-Thesis/FIG_CollisionOperator"
output_base_path = "/home/mbille/lettuce/plotting_MA"
plot_batch_name = "test_LinePlotting_CO"

if not os.path.exists(output_base_path + "/" + plot_batch_name):
    os.makedirs(output_base_path + "/" + plot_batch_name)

co_variants = ["BGK", "KBC", "REG"]
timesteps = [10000, 30000]
bc_variants = ["noBBBC", "IBBd0.5"]
yrel_variants = ["rel0", "yrel5"]
# CSV format:
# "TimeStep", "p" ,"ux" ,"uy" ,"uz" ,"Points:0" ,"Points:1" ,"Points:2"
# 0            1    2     3     4     5   x        6   y        7  z

if False:

    ### FIG_CO_1: KBC-oscillation around i10000, reg does not.
    data_fig_path = "/lowInlet_tempOsz_step10000_REGvsKBC_ibb05"

    # READ DATA
    timesteps = [10000, 10001, 10002, 10003]
    yrels = [0, 1]

    # read and uniqui-fy datasets
    data_reg_ibb05_low_step10000_yrel0 = np.unique(np.genfromtxt(data_base_path + data_fig_path + "/REG_ibbd05_low_step10000_yrel0.csv", delimiter=",", skip_header=1), axis=0)
    data_reg_ibb05_low_step10000_yrel4 = np.unique(np.genfromtxt(data_base_path + data_fig_path + "/REG_ibbd05_low_step10000_yrel4.csv", delimiter=",", skip_header=1), axis=0)

    data_kbc_ibbd05_low_step10000_yrel0 = np.unique(np.genfromtxt(data_base_path + data_fig_path + "/KBC_ibbd05_low_step10000_yrel0.csv", delimiter=",", skip_header=1), axis=0)
    data_kbc_ibbd05_low_step10000_yrel4 = np.unique(np.genfromtxt(data_base_path + data_fig_path + "/KBC_ibbd05_low_step10000_yrel4.csv", delimiter=",", skip_header=1), axis=0)
    data_kbc_ibbd05_low_step10001_yrel0 = np.unique(np.genfromtxt(data_base_path + data_fig_path + "/KBC_ibbd05_low_step10001_yrel0.csv", delimiter=",", skip_header=1), axis=0)
    data_kbc_ibbd05_low_step10001_yrel4 = np.unique(np.genfromtxt(data_base_path + data_fig_path + "/KBC_ibbd05_low_step10001_yrel4.csv", delimiter=",", skip_header=1), axis=0)
    data_kbc_ibbd05_low_step10002_yrel0 = np.unique(np.genfromtxt(data_base_path + data_fig_path + "/KBC_ibbd05_low_step10002_yrel0.csv", delimiter=",", skip_header=1), axis=0)
    data_kbc_ibbd05_low_step10002_yrel4 = np.unique(np.genfromtxt(data_base_path + data_fig_path + "/KBC_ibbd05_low_step10002_yrel4.csv", delimiter=",", skip_header=1), axis=0)
    data_kbc_ibbd05_low_step10003_yrel0 = np.unique(np.genfromtxt(data_base_path + data_fig_path + "/KBC_ibbd05_low_step10003_yrel0.csv", delimiter=",", skip_header=1), axis=0)
    data_kbc_ibbd05_low_step10003_yrel4 = np.unique(np.genfromtxt(data_base_path + data_fig_path + "/KBC_ibbd05_low_step10003_yrel4.csv", delimiter=",", skip_header=1), axis=0)

    # THIS SOMEHOW DOESN't WORK:...
    # datasets = [data_reg_ibb05_low_step10000_yrel0, data_reg_ibb05_low_step10000_yrel4,
    #             data_kbc_ibbd05_low_step10000_yrel0, data_kbc_ibbd05_low_step10000_yrel4,
    #             data_kbc_ibbd05_low_step10001_yrel0, data_kbc_ibbd05_low_step10001_yrel4,
    #             data_kbc_ibbd05_low_step10002_yrel0, data_kbc_ibbd05_low_step10002_yrel4,
    #             data_kbc_ibbd05_low_step10003_yrel0, data_kbc_ibbd05_low_step10003_yrel4]
    # for dataset in datasets:
    #     sort_indices = np.argsort(dataset[:, 5])
    #     dataset = dataset[sort_indices]

    sort_indices = np.argsort(data_reg_ibb05_low_step10000_yrel0[:, 5])
    data_reg_ibb05_low_step10000_yrel0 = data_reg_ibb05_low_step10000_yrel0[sort_indices]

    sort_indices = np.argsort(data_reg_ibb05_low_step10000_yrel4[:, 5])
    data_reg_ibb05_low_step10000_yrel4 = data_reg_ibb05_low_step10000_yrel4[sort_indices]

    sort_indices = np.argsort(data_kbc_ibbd05_low_step10000_yrel0[:, 5])
    data_kbc_ibbd05_low_step10000_yrel0 = data_kbc_ibbd05_low_step10000_yrel0[sort_indices]

    sort_indices = np.argsort(data_kbc_ibbd05_low_step10000_yrel4[:, 5])
    data_kbc_ibbd05_low_step10000_yrel4 = data_kbc_ibbd05_low_step10000_yrel4[sort_indices]

    sort_indices = np.argsort(data_kbc_ibbd05_low_step10001_yrel0[:, 5])
    data_kbc_ibbd05_low_step10001_yrel0 = data_kbc_ibbd05_low_step10001_yrel0[sort_indices]

    sort_indices = np.argsort(data_kbc_ibbd05_low_step10001_yrel4[:, 5])
    data_kbc_ibbd05_low_step10001_yrel4 = data_kbc_ibbd05_low_step10001_yrel4[sort_indices]

    sort_indices = np.argsort(data_kbc_ibbd05_low_step10002_yrel0[:, 5])
    data_kbc_ibbd05_low_step10002_yrel0 = data_kbc_ibbd05_low_step10002_yrel0[sort_indices]

    sort_indices = np.argsort(data_kbc_ibbd05_low_step10002_yrel4[:, 5])
    data_kbc_ibbd05_low_step10002_yrel4 = data_kbc_ibbd05_low_step10002_yrel4[sort_indices]

    sort_indices = np.argsort(data_kbc_ibbd05_low_step10003_yrel0[:, 5])
    data_kbc_ibbd05_low_step10003_yrel0 = data_kbc_ibbd05_low_step10003_yrel0[sort_indices]

    sort_indices = np.argsort(data_kbc_ibbd05_low_step10003_yrel4[:, 5])
    data_kbc_ibbd05_low_step10003_yrel4 = data_kbc_ibbd05_low_step10003_yrel4[sort_indices]

    fig_co_1, axs_co = plt.subplots(3,2, sharex='col', sharey='row')
    # p, ux, uy (3 observables)  -> 3 rows
    # reg + 3 timesteps (4 lines per plot) -> 4 LINES PER PLOT
    # 2 positions... yrel -> 2 COLS

    # p @ yrel0
    axs_co[0,0].plot(data_reg_ibb05_low_step10000_yrel0[:,5],data_reg_ibb05_low_step10000_yrel0[:,1],
                       data_kbc_ibbd05_low_step10000_yrel0[:,5],data_kbc_ibbd05_low_step10000_yrel0[:,1],
                       data_kbc_ibbd05_low_step10001_yrel0[:,5], data_kbc_ibbd05_low_step10001_yrel0[:,1],
                       data_kbc_ibbd05_low_step10002_yrel0[:,5], data_kbc_ibbd05_low_step10002_yrel0[:,1],
                       data_kbc_ibbd05_low_step10003_yrel0[:,5], data_kbc_ibbd05_low_step10003_yrel0[:,1],
                       )

    # p @ yrel4
    axs_co[0,1].plot(data_reg_ibb05_low_step10000_yrel4[:,5],data_reg_ibb05_low_step10000_yrel4[:,1],
                       data_kbc_ibbd05_low_step10000_yrel4[:,5],data_kbc_ibbd05_low_step10000_yrel4[:,1],
                       data_kbc_ibbd05_low_step10001_yrel4[:,5], data_kbc_ibbd05_low_step10001_yrel4[:,1],
                       data_kbc_ibbd05_low_step10002_yrel4[:,5], data_kbc_ibbd05_low_step10002_yrel4[:,1],
                       data_kbc_ibbd05_low_step10003_yrel4[:,5], data_kbc_ibbd05_low_step10003_yrel4[:,1],
                       )

    # ux @yrel0
    axs_co[1,0].plot(data_reg_ibb05_low_step10000_yrel0[:,5],data_reg_ibb05_low_step10000_yrel0[:,2],
                       data_kbc_ibbd05_low_step10000_yrel0[:,5],data_kbc_ibbd05_low_step10000_yrel0[:,2],
                       data_kbc_ibbd05_low_step10001_yrel0[:,5], data_kbc_ibbd05_low_step10001_yrel0[:,2],
                       data_kbc_ibbd05_low_step10002_yrel0[:,5], data_kbc_ibbd05_low_step10002_yrel0[:,2],
                       data_kbc_ibbd05_low_step10003_yrel0[:,5], data_kbc_ibbd05_low_step10003_yrel0[:,2],
                       )

    # ux @yrel4
    axs_co[1,1].plot(data_reg_ibb05_low_step10000_yrel4[:,5],data_reg_ibb05_low_step10000_yrel4[:,2],
                       data_kbc_ibbd05_low_step10000_yrel4[:,5],data_kbc_ibbd05_low_step10000_yrel4[:,2],
                       data_kbc_ibbd05_low_step10001_yrel4[:,5], data_kbc_ibbd05_low_step10001_yrel4[:,2],
                       data_kbc_ibbd05_low_step10002_yrel4[:,5], data_kbc_ibbd05_low_step10002_yrel4[:,2],
                       data_kbc_ibbd05_low_step10003_yrel4[:,5], data_kbc_ibbd05_low_step10003_yrel4[:,2],
                       )

    # uy @yrel0
    axs_co[2,0].plot(data_reg_ibb05_low_step10000_yrel0[:,5],data_reg_ibb05_low_step10000_yrel0[:,3],
                       data_kbc_ibbd05_low_step10000_yrel0[:,5],data_kbc_ibbd05_low_step10000_yrel0[:,3],
                       data_kbc_ibbd05_low_step10001_yrel0[:,5], data_kbc_ibbd05_low_step10001_yrel0[:,3],
                       data_kbc_ibbd05_low_step10002_yrel0[:,5], data_kbc_ibbd05_low_step10002_yrel0[:,3],
                       data_kbc_ibbd05_low_step10003_yrel0[:,5], data_kbc_ibbd05_low_step10003_yrel0[:,3],
                       )

    # uy @urel4
    axs_co[2,1].plot(data_reg_ibb05_low_step10000_yrel4[:,5],data_reg_ibb05_low_step10000_yrel4[:,3],
                       data_kbc_ibbd05_low_step10000_yrel4[:,5],data_kbc_ibbd05_low_step10000_yrel4[:,3],
                       data_kbc_ibbd05_low_step10001_yrel4[:,5], data_kbc_ibbd05_low_step10001_yrel4[:,3],
                       data_kbc_ibbd05_low_step10002_yrel4[:,5], data_kbc_ibbd05_low_step10002_yrel4[:,3],
                       data_kbc_ibbd05_low_step10003_yrel4[:,5], data_kbc_ibbd05_low_step10003_yrel4[:,3],
                       )

    # x-axis labels
    axs_co[2,0].set_xlabel(r"$x_{LU}$")
    axs_co[2,1].set_xlabel(r"$x_{LU}$")

    # y-axis labels
    axs_co[0,0].set_ylabel(r"$p_{LU}$")
    axs_co[1,0].set_ylabel(r"$u_{x,LU}$")
    axs_co[2,0].set_ylabel(r"$u_{y,LU}$")

    # y-axis limits
    axs_co[0,0].set_ylim([-0.0001,0.0001])
    axs_co[1,0].set_ylim([-0.015,0.015])
    axs_co[2,0].set_ylim([-0.015,0.015])

    # x-axis limits
    axs_co[2,0].set_xlim([0,40])
    axs_co[2,1].set_xlim([0,40])

    axs_co[0,0].set_title(r"at $y_{LU} = 1$")
    axs_co[0,1].set_title(r"at $y_{LU} = 5$")

    axs_co[2,1].legend(labels=["REG","KBC, i=10000","KBC, i=10001","KBC, i=10002","KBC, i=10003"],fontsize=6)
    #TODO: add subtitle for ax[0,1] and ax[0,0] to be yrel0 and yrel4 (!)
    plt.suptitle(f"p/ux/uy(x,ti), REG vs. KBC around ti=10000, RES8, lowInlet")
    plt.savefig(output_base_path + "/" + plot_batch_name + "/" + "test_lwInlet_CO_IBB")
    plt.close(fig_co_1)

####################################

### FIG_CO_2: KBC temporal oscillation around ti30000+3:
# noBBBC, KBC, yrelMID (10, 15)
data_fig_path = "/tempOsz_mid_step30000_KBC_andCrash_noBBBC"

# READ DATA
timesteps = [10000, 10001, 10002, 10003]
yrels = [0, 5]

data_kbc_noBBBC_low_step30000_yrel0 = np.unique(np.genfromtxt(data_base_path + data_fig_path + "/KBC_step30000_yrel0.csv", delimiter=",", skip_header=1), axis=0)
data_kbc_noBBBC_low_step30000_yrel5 = np.unique(np.genfromtxt(data_base_path + data_fig_path + "/KBC_step30000_yrel5.csv", delimiter=",", skip_header=1), axis=0)
data_kbc_noBBBC_low_step30001_yrel0 = np.unique(np.genfromtxt(data_base_path + data_fig_path + "/KBC_step30001_yrel0.csv", delimiter=",", skip_header=1), axis=0)
data_kbc_noBBBC_low_step30001_yrel5 = np.unique(np.genfromtxt(data_base_path + data_fig_path + "/KBC_step30001_yrel5.csv", delimiter=",", skip_header=1), axis=0)
data_kbc_noBBBC_low_step30002_yrel0 = np.unique(np.genfromtxt(data_base_path + data_fig_path + "/KBC_step30002_yrel0.csv", delimiter=",", skip_header=1), axis=0)
data_kbc_noBBBC_low_step30002_yrel5 = np.unique(np.genfromtxt(data_base_path + data_fig_path + "/KBC_step30002_yrel5.csv", delimiter=",", skip_header=1), axis=0)
data_kbc_noBBBC_low_step30003_yrel0 = np.unique(np.genfromtxt(data_base_path + data_fig_path + "/KBC_step30003_yrel0.csv", delimiter=",", skip_header=1), axis=0)
data_kbc_noBBBC_low_step30003_yrel5 = np.unique(np.genfromtxt(data_base_path + data_fig_path + "/KBC_step30003_yrel5.csv", delimiter=",", skip_header=1), axis=0)

sort_indices = np.argsort(data_kbc_noBBBC_low_step30000_yrel0[:, 5])
data_kbc_noBBBC_low_step30000_yrel0 = data_kbc_noBBBC_low_step30000_yrel0[sort_indices]

sort_indices = np.argsort(data_kbc_noBBBC_low_step30000_yrel5[:, 5])
data_kbc_noBBBC_low_step30000_yrel5 = data_kbc_noBBBC_low_step30000_yrel5[sort_indices]

sort_indices = np.argsort(data_kbc_noBBBC_low_step30001_yrel0[:, 5])
data_kbc_noBBBC_low_step30001_yrel0 = data_kbc_noBBBC_low_step30001_yrel0[sort_indices]

sort_indices = np.argsort(data_kbc_noBBBC_low_step30001_yrel5[:, 5])
data_kbc_noBBBC_low_step30001_yrel5 = data_kbc_noBBBC_low_step30001_yrel5[sort_indices]

sort_indices = np.argsort(data_kbc_noBBBC_low_step30002_yrel0[:, 5])
data_kbc_noBBBC_low_step30002_yrel0 = data_kbc_noBBBC_low_step30002_yrel0[sort_indices]

sort_indices = np.argsort(data_kbc_noBBBC_low_step30002_yrel5[:, 5])
data_kbc_noBBBC_low_step30002_yrel5 = data_kbc_noBBBC_low_step30002_yrel5[sort_indices]

sort_indices = np.argsort(data_kbc_noBBBC_low_step30003_yrel0[:, 5])
data_kbc_noBBBC_low_step30003_yrel0 = data_kbc_noBBBC_low_step30003_yrel0[sort_indices]

sort_indices = np.argsort(data_kbc_noBBBC_low_step30003_yrel5[:, 5])
data_kbc_noBBBC_low_step30003_yrel5 = data_kbc_noBBBC_low_step30003_yrel5[sort_indices]

fig_co_2, axs_co = plt.subplots(3,2, sharex='col', sharey='row')
# p, ux, uy (3 observables)  -> 3 rows
# reg + 3 timesteps (4 lines per plot) -> 4 LINES PER PLOT
# 2 positions... yrel -> 2 COLS

# p @ yrel0
axs_co[0,0].plot(data_kbc_noBBBC_low_step30000_yrel0[:,5],data_kbc_noBBBC_low_step30000_yrel0[:,1],
                   data_kbc_noBBBC_low_step30001_yrel0[:,5], data_kbc_noBBBC_low_step30001_yrel0[:,1],
                   data_kbc_noBBBC_low_step30002_yrel0[:,5], data_kbc_noBBBC_low_step30002_yrel0[:,1],
                   data_kbc_noBBBC_low_step30003_yrel0[:,5], data_kbc_noBBBC_low_step30003_yrel0[:,1]
                 )

# p @ yrel5
axs_co[0,1].plot(data_kbc_noBBBC_low_step30000_yrel5[:,5],data_kbc_noBBBC_low_step30000_yrel5[:,1],
                   data_kbc_noBBBC_low_step30001_yrel5[:,5], data_kbc_noBBBC_low_step30001_yrel5[:,1],
                   data_kbc_noBBBC_low_step30002_yrel5[:,5], data_kbc_noBBBC_low_step30002_yrel5[:,1],
                   data_kbc_noBBBC_low_step30003_yrel5[:,5], data_kbc_noBBBC_low_step30003_yrel5[:,1]
                   )

# ux @yrel0
axs_co[1,0].plot(data_kbc_noBBBC_low_step30000_yrel0[:,5],data_kbc_noBBBC_low_step30000_yrel0[:,2],
                   data_kbc_noBBBC_low_step30001_yrel0[:,5], data_kbc_noBBBC_low_step30001_yrel0[:,2],
                   data_kbc_noBBBC_low_step30002_yrel0[:,5], data_kbc_noBBBC_low_step30002_yrel0[:,2],
                   data_kbc_noBBBC_low_step30003_yrel0[:,5], data_kbc_noBBBC_low_step30003_yrel0[:,2]
                   )

# ux @yrel5
axs_co[1,1].plot(data_kbc_noBBBC_low_step30000_yrel5[:,5],data_kbc_noBBBC_low_step30000_yrel5[:,2],
                   data_kbc_noBBBC_low_step30001_yrel5[:,5], data_kbc_noBBBC_low_step30001_yrel5[:,2],
                   data_kbc_noBBBC_low_step30002_yrel5[:,5], data_kbc_noBBBC_low_step30002_yrel5[:,2],
                   data_kbc_noBBBC_low_step30003_yrel5[:,5], data_kbc_noBBBC_low_step30003_yrel5[:,2]
                   )

# uy @yrel0
axs_co[2,0].plot(data_kbc_noBBBC_low_step30000_yrel0[:,5],data_kbc_noBBBC_low_step30000_yrel0[:,3],
                   data_kbc_noBBBC_low_step30001_yrel0[:,5], data_kbc_noBBBC_low_step30001_yrel0[:,3],
                   data_kbc_noBBBC_low_step30002_yrel0[:,5], data_kbc_noBBBC_low_step30002_yrel0[:,3],
                   data_kbc_noBBBC_low_step30003_yrel0[:,5], data_kbc_noBBBC_low_step30003_yrel0[:,3]
                   )

# uy @urel5
axs_co[2,1].plot(data_kbc_noBBBC_low_step30000_yrel5[:,5],data_kbc_noBBBC_low_step30000_yrel5[:,3],
                   data_kbc_noBBBC_low_step30001_yrel5[:,5], data_kbc_noBBBC_low_step30001_yrel5[:,3],
                   data_kbc_noBBBC_low_step30002_yrel5[:,5], data_kbc_noBBBC_low_step30002_yrel5[:,3],
                   data_kbc_noBBBC_low_step30003_yrel5[:,5], data_kbc_noBBBC_low_step30003_yrel5[:,3]
                   )

# x-axis labels
axs_co[2,0].set_xlabel(r"$x_{LU}$")
axs_co[2,1].set_xlabel(r"$x_{LU}$")

# y-axis labels
axs_co[0,0].set_ylabel(r"$p_{LU}$")
axs_co[1,0].set_ylabel(r"$u_{x,LU}$")
axs_co[2,0].set_ylabel(r"$u_{y,LU}$")

# y-axis limits
axs_co[0,0].set_ylim([-0.0001,0.0001])
axs_co[1,0].set_ylim([-0.005,0.005])
axs_co[2,0].set_ylim([-0.005,0.005])

# x-axis limits
axs_co[2,0].set_xlim([0,80])
axs_co[2,1].set_xlim([0,80])

axs_co[0,0].set_title(r"at $y_{LU} = 10$ (Inlet node height)")
axs_co[0,1].set_title(r"at $y_{LU} = 15$ (Inlet node height + 5 nodes)")

axs_co[2,1].legend(labels=["KBC, i=30000","KBC, i=30001","KBC, i=30002","KBC, i=30003"],fontsize=6)
#TODO: add subtitle for ax[0,1] and ax[0,0] to be yrel0 and yrel4 (!)
plt.suptitle(f"p/ux/uy(x,ti), KBC, noBBBC around ti=30000, RES8, MidInlet")
plt.savefig(output_base_path + "/" + plot_batch_name + "/" + "test_midInlet_KBC_noBBBC_tempOsc")
plt.close(fig_co_2)