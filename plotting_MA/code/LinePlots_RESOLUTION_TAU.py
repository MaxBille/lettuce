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
matplotlib.rcParams.update({'lines.linewidth': 1})
#matplotlib.rcParams.update({'lines.linestyle': '--'})
matplotlib.rcParams.update({'font.size': 8}) # font size was 11
matplotlib.rcParams.update({'figure.figsize': (7.22433,6)})

### DATA I/O settings
data_base_path = "/home/mbille/Desktop/MA_Paraview_DataAndScreenshots-for-Thesis/FIG_tau_res/t1944 oder step 14400"
output_base_path = "/home/mbille/lettuce/plotting_MA"
plot_batch_name = "test_LinePlotting_TAU"

if not os.path.exists(output_base_path + "/" + plot_batch_name):
    os.makedirs(output_base_path + "/" + plot_batch_name)

#KBC co_variants = ["BGK", "KBC", "REG"]

# variants...timesteps = [10000, 30000]

# RESOLUTION    8       100     1000
# TIMESTEP      30.000  375.000 3.750.000
# T_PU          4.164,819 s

# ibbd05 bc_variants = ["noBBBC", "IBBd0.5"]
yrel_variants = ["rel0", "yrel5"]

# CSV format:
# "TimeStep", "p" ,"ux" ,"uy" ,"uz" ,"Points:0" ,"Points:1" ,"Points:2"
# 0            1    2     3     4     5   x        6   y        7  z

# NEW:
# "p" ,"ux" ,"uy" ,"uz" ,"Points:0" ,"Points:1" ,"Points:2"
#  0    1     2     3     4   x        5    y       6  z


### fig_res_1: spacial resolution and thereby tau influences the artifact-severity
data_fig_path = ""

# READ DATA
timesteps = [10000, 10001, 10002, 10003]
yrels = [0, 5]

# read and uniqui-fy datasets
# (!) the first two datasets have no timeStep in the table...
# TIME PU ~1944, and equivalent step for all sims (expect res50)

def load_data(path, has_extra_col=False):
    """
    Lädt die Daten und entfernt die erste Spalte, wenn der Datensatz eine zusätzliche Spalte hat.
    """
    data = np.unique(np.genfromtxt(path, delimiter=",", skip_header=1), axis=0)
    if has_extra_col:
        data = data[:, 1:]  # Entfernt die erste Spalte, wenn extra_col True ist
    return data

# Einlesen der Daten
data_kbc_ibbd05_res8_low_yrel0 = load_data(data_base_path + data_fig_path + "/kbc_ibbd05_res8_low_step14000_t1944_yrel0.csv")
data_kbc_ibbd05_res8_low_yrel5 = load_data(data_base_path + data_fig_path + "/kbc_ibbd05_res8_low_step14000_t1944_yrel5.csv")
data_kbc_ibbd05_res16_low_yrel0 = load_data(data_base_path + data_fig_path + "/kbc_ibbd05_res16_low_step28000_tY_yrel0.csv", has_extra_col=True)
data_kbc_ibbd05_res16_low_yrel5 = load_data(data_base_path + data_fig_path + "/kbc_ibbd05_res16_low_step28000_tY_yrel5.csv", has_extra_col=True)
data_kbc_ibbd05_res50_low_yrel0 = load_data(data_base_path + data_fig_path + "/kbc_ibbd05_res50_low_step61000_tYaltCRASH_yrel0.csv", has_extra_col=True)
data_kbc_ibbd05_res50_low_yrel5 = load_data(data_base_path + data_fig_path + "/kbc_ibbd05_res50_low_step61000_tYaltCRASH_yrel5.csv", has_extra_col=True)
data_kbc_ibbd05_res100_low_yrel0 = load_data(data_base_path + data_fig_path + "/kbc_ibbd05_res100_low_step175000_tY_yrel0.csv", has_extra_col=True)
data_kbc_ibbd05_res100_low_yrel5 = load_data(data_base_path + data_fig_path + "/kbc_ibbd05_res100_low_step175000_tY_yrel5.csv", has_extra_col=True)
data_kbc_ibbd05_res1000_low_yrel0 = load_data(data_base_path + data_fig_path + "/kbc_ibbd05_res1000_low_step1750000_tY_yrel0.csv", has_extra_col=True)
data_kbc_ibbd05_res1000_low_yrel5 = load_data(data_base_path + data_fig_path + "/kbc_ibbd05_res1000_low_step1750000_tY_yrel5.csv", has_extra_col=True)

# data_kbc_ibbd05_res8_low_yrel0 = np.unique(np.genfromtxt(data_base_path + data_fig_path + "/kbc_ibbd05_res8_low_step14000_t1944_yrel0.csv", delimiter=",", skip_header=1), axis=0)
# data_kbc_ibbd05_res8_low_yrel5 = np.unique(np.genfromtxt(data_base_path + data_fig_path + "/kbc_ibbd05_res8_low_step14000_t1944_yrel5.csv", delimiter=",", skip_header=1), axis=0)
# data_kbc_ibbd05_res16_low_yrel0 = np.unique(np.genfromtxt(data_base_path + data_fig_path + "/kbc_ibbd05_res16_low_step28000_tY_yrel0.csv", delimiter=",", skip_header=1), axis=0)
# data_kbc_ibbd05_res16_low_yrel5 = np.unique(np.genfromtxt(data_base_path + data_fig_path + "/kbc_ibbd05_res16_low_step28000_tY_yrel5.csv", delimiter=",", skip_header=1), axis=0)
# data_kbc_ibbd05_res50_low_yrel0 = np.unique(np.genfromtxt(data_base_path + data_fig_path + "/kbc_ibbd05_res50_low_step61000_tYaltCRASH_yrel0.csv", delimiter=",", skip_header=1), axis=0)
# data_kbc_ibbd05_res50_low_yrel5 = np.unique(np.genfromtxt(data_base_path + data_fig_path + "/kbc_ibbd05_res50_low_step61000_tYaltCRASH_yrel5.csv", delimiter=",", skip_header=1), axis=0)
# data_kbc_ibbd05_res100_low_yrel0 = np.unique(np.genfromtxt(data_base_path + data_fig_path + "/kbc_ibbd05_res100_low_step175000_tY_yrel0.csv", delimiter=",", skip_header=1), axis=0)
# data_kbc_ibbd05_res100_low_yrel5 = np.unique(np.genfromtxt(data_base_path + data_fig_path + "/kbc_ibbd05_res100_low_step175000_tY_yrel5.csv", delimiter=",", skip_header=1), axis=0)
# data_kbc_ibbd05_res1000_low_yrel0 = np.unique(np.genfromtxt(data_base_path + data_fig_path + "/kbc_ibbd05_res1000_low_step1750000_tY_yrel0.csv", delimiter=",", skip_header=1), axis=0)
# data_kbc_ibbd05_res1000_low_yrel5 = np.unique(np.genfromtxt(data_base_path + data_fig_path + "/kbc_ibbd05_res1000_low_step1750000_tY_yrel5.csv", delimiter=",", skip_header=1), axis=0)

# Mapping von Variablennamen zu tatsächlichen Datenobjekten
data_sets = {
    "res8_yrel0": data_kbc_ibbd05_res8_low_yrel0,
    "res8_yrel5": data_kbc_ibbd05_res8_low_yrel5,
    "res16_yrel0": data_kbc_ibbd05_res16_low_yrel0,
    "res16_yrel5": data_kbc_ibbd05_res16_low_yrel5,
    "res50_yrel0": data_kbc_ibbd05_res50_low_yrel0,
    "res50_yrel5": data_kbc_ibbd05_res50_low_yrel5,
    "res100_yrel0": data_kbc_ibbd05_res100_low_yrel0,
    "res100_yrel5": data_kbc_ibbd05_res100_low_yrel5,
    "res1000_yrel0": data_kbc_ibbd05_res1000_low_yrel0,
    "res1000_yrel5": data_kbc_ibbd05_res1000_low_yrel5,
}

# Sortiere alle Datensätze nach Spalte 4
for key, data in data_sets.items():
    sort_idx = np.argsort(data[:, 4])
    data_sets[key] = data[sort_idx]

# Optional: zurück in ursprüngliche Variablen schreiben
data_kbc_ibbd05_res8_low_yrel0    = data_sets["res8_yrel0"]
data_kbc_ibbd05_res8_low_yrel5    = data_sets["res8_yrel5"]
data_kbc_ibbd05_res16_low_yrel0   = data_sets["res16_yrel0"]
data_kbc_ibbd05_res16_low_yrel5   = data_sets["res16_yrel5"]
data_kbc_ibbd05_res50_low_yrel0   = data_sets["res50_yrel0"]
data_kbc_ibbd05_res50_low_yrel5   = data_sets["res50_yrel5"]
data_kbc_ibbd05_res100_low_yrel0  = data_sets["res100_yrel0"]
data_kbc_ibbd05_res100_low_yrel5  = data_sets["res100_yrel5"]
data_kbc_ibbd05_res1000_low_yrel0 = data_sets["res1000_yrel0"]
data_kbc_ibbd05_res1000_low_yrel5 = data_sets["res1000_yrel5"]


fig_res_1, axs_res = plt.subplots(3, 2, sharex='col', sharey='row')

# --- p @ yrel0 ---
axs_res[0, 0].plot(
    data_kbc_ibbd05_res8_low_yrel0[:, 4], data_kbc_ibbd05_res8_low_yrel0[:, 0],
    data_kbc_ibbd05_res16_low_yrel0[:, 4], data_kbc_ibbd05_res16_low_yrel0[:, 0],
    data_kbc_ibbd05_res50_low_yrel0[:, 4], data_kbc_ibbd05_res50_low_yrel0[:, 0],
    data_kbc_ibbd05_res100_low_yrel0[:, 4], data_kbc_ibbd05_res100_low_yrel0[:, 0],
    data_kbc_ibbd05_res1000_low_yrel0[:, 4], data_kbc_ibbd05_res1000_low_yrel0[:, 0]
)
print(data_kbc_ibbd05_res16_low_yrel0[:5])
print("res16 yrel0:", data_kbc_ibbd05_res16_low_yrel0.shape)
print("res50 yrel0:", data_kbc_ibbd05_res50_low_yrel0.shape)
print("res8 yrel0:", data_kbc_ibbd05_res8_low_yrel0.shape)

# --- p @ yrel5 ---
axs_res[0, 1].plot(
    data_kbc_ibbd05_res8_low_yrel5[:, 4], data_kbc_ibbd05_res8_low_yrel5[:, 0],
    data_kbc_ibbd05_res16_low_yrel5[:, 4], data_kbc_ibbd05_res16_low_yrel5[:, 0],
    data_kbc_ibbd05_res50_low_yrel5[:, 4], data_kbc_ibbd05_res50_low_yrel5[:, 0],
    data_kbc_ibbd05_res100_low_yrel5[:, 4], data_kbc_ibbd05_res100_low_yrel5[:, 0],
    data_kbc_ibbd05_res1000_low_yrel5[:, 4], data_kbc_ibbd05_res1000_low_yrel5[:, 0]
)

# --- ux @ yrel0 ---
axs_res[1, 0].plot(
    data_kbc_ibbd05_res8_low_yrel0[:, 4], data_kbc_ibbd05_res8_low_yrel0[:, 1],
    data_kbc_ibbd05_res16_low_yrel0[:, 4], data_kbc_ibbd05_res16_low_yrel0[:, 1],
    data_kbc_ibbd05_res50_low_yrel0[:, 4], data_kbc_ibbd05_res50_low_yrel0[:, 1],
    data_kbc_ibbd05_res100_low_yrel0[:, 4], data_kbc_ibbd05_res100_low_yrel0[:, 1],
    data_kbc_ibbd05_res1000_low_yrel0[:, 4], data_kbc_ibbd05_res1000_low_yrel0[:, 1]
)

# --- ux @ yrel5 ---
axs_res[1, 1].plot(
    data_kbc_ibbd05_res8_low_yrel5[:, 4], data_kbc_ibbd05_res8_low_yrel5[:, 1],
    data_kbc_ibbd05_res16_low_yrel5[:, 4], data_kbc_ibbd05_res16_low_yrel5[:, 1],
    data_kbc_ibbd05_res50_low_yrel5[:, 4], data_kbc_ibbd05_res50_low_yrel5[:, 1],
    data_kbc_ibbd05_res100_low_yrel5[:, 4], data_kbc_ibbd05_res100_low_yrel5[:, 1],
    data_kbc_ibbd05_res1000_low_yrel5[:, 4], data_kbc_ibbd05_res1000_low_yrel5[:, 1]
)

# --- uy @ yrel0 ---
axs_res[2, 0].plot(
    data_kbc_ibbd05_res8_low_yrel0[:, 4], data_kbc_ibbd05_res8_low_yrel0[:, 2],
    data_kbc_ibbd05_res16_low_yrel0[:, 4], data_kbc_ibbd05_res16_low_yrel0[:, 2],
    data_kbc_ibbd05_res50_low_yrel0[:, 4], data_kbc_ibbd05_res50_low_yrel0[:, 2],
    data_kbc_ibbd05_res100_low_yrel0[:, 4], data_kbc_ibbd05_res100_low_yrel0[:, 2],
    data_kbc_ibbd05_res1000_low_yrel0[:, 4], data_kbc_ibbd05_res1000_low_yrel0[:, 2]
)

# --- uy @ yrel5 ---
axs_res[2, 1].plot(
    data_kbc_ibbd05_res8_low_yrel5[:, 4], data_kbc_ibbd05_res8_low_yrel5[:, 2],
    data_kbc_ibbd05_res16_low_yrel5[:, 4], data_kbc_ibbd05_res16_low_yrel5[:, 2],
    data_kbc_ibbd05_res50_low_yrel5[:, 4], data_kbc_ibbd05_res50_low_yrel5[:, 2],
    data_kbc_ibbd05_res100_low_yrel5[:, 4], data_kbc_ibbd05_res100_low_yrel5[:, 2],
    data_kbc_ibbd05_res1000_low_yrel5[:, 4], data_kbc_ibbd05_res1000_low_yrel5[:, 2]
)

# x-axis labels
axs_res[2,0].set_xlabel(r"$x_{LU}$")
axs_res[2,1].set_xlabel(r"$x_{LU}$")

# y-axis labels
axs_res[0,0].set_ylabel(r"$p$")
axs_res[1,0].set_ylabel(r"$u_{x}$")
axs_res[2,0].set_ylabel(r"$u_{y}$")

# y-axis limits
axs_res[0,0].set_ylim([-0.001,0.001])
axs_res[1,0].set_ylim([-0.02,0.02])
axs_res[2,0].set_ylim([-0.02,0.02])

# x-axis limits
axs_res[2,0].set_xlim([0,40])
axs_res[2,1].set_xlim([0,40])

axs_res[0,0].set_title(r"Position $y_{LU} = 1$")
axs_res[0,1].set_title(r"Position $y_{LU} = 5$")

axs_res[2,1].legend(labels=["8 LU/PU", "16 LU/PU", "50 LU/PU", "100 LU/PU", "1000 LU/PU"],fontsize=6)

#plt.suptitle(f"p/ux/uy(x,ti), KBC, ibbd05, lowInlet, RES (8,100,1000), t_PU=4165 (30000steps for res8), res8 at i14000 (crash afterwards)")
plt.savefig(output_base_path + "/" + plot_batch_name + "/" + "lowInlet_RES_TAU_IBB_allRES_puxuy")
plt.close(fig_res_1)

####################################


### OHNE uy:

matplotlib.rcParams.update({'figure.figsize': (7.22433,4)})
fig_res_2, axs_res = plt.subplots(2, 2, sharex='col', sharey='row')

# p @ yrel0
axs_res[0, 0].plot(data_kbc_ibbd05_res8_low_yrel0[:, 4], data_kbc_ibbd05_res8_low_yrel0[:, 0],
                   data_kbc_ibbd05_res16_low_yrel0[:, 4], data_kbc_ibbd05_res16_low_yrel0[:, 0],
                   data_kbc_ibbd05_res50_low_yrel0[:, 4], data_kbc_ibbd05_res50_low_yrel0[:, 0],
                   data_kbc_ibbd05_res100_low_yrel0[:, 4], data_kbc_ibbd05_res100_low_yrel0[:, 0],
                   data_kbc_ibbd05_res1000_low_yrel0[:, 4], data_kbc_ibbd05_res1000_low_yrel0[:, 0])

# p @ yrel5
axs_res[0, 1].plot(data_kbc_ibbd05_res8_low_yrel5[:, 4], data_kbc_ibbd05_res8_low_yrel5[:, 0],
                   data_kbc_ibbd05_res16_low_yrel5[:, 4], data_kbc_ibbd05_res16_low_yrel5[:, 0],
                   data_kbc_ibbd05_res50_low_yrel5[:, 4], data_kbc_ibbd05_res50_low_yrel5[:, 0],
                   data_kbc_ibbd05_res100_low_yrel5[:, 4], data_kbc_ibbd05_res100_low_yrel5[:, 0],
                   data_kbc_ibbd05_res1000_low_yrel5[:, 4], data_kbc_ibbd05_res1000_low_yrel5[:, 0])

# ux @ yrel0
axs_res[1, 0].plot(data_kbc_ibbd05_res8_low_yrel0[:, 4], data_kbc_ibbd05_res8_low_yrel0[:, 1],
                   data_kbc_ibbd05_res16_low_yrel0[:, 4], data_kbc_ibbd05_res16_low_yrel0[:, 1],
                   data_kbc_ibbd05_res50_low_yrel0[:, 4], data_kbc_ibbd05_res50_low_yrel0[:, 1],
                   data_kbc_ibbd05_res100_low_yrel0[:, 4], data_kbc_ibbd05_res100_low_yrel0[:, 1],
                   data_kbc_ibbd05_res1000_low_yrel0[:, 4], data_kbc_ibbd05_res1000_low_yrel0[:, 1])

# ux @ yrel5
axs_res[1, 1].plot(data_kbc_ibbd05_res8_low_yrel5[:, 4], data_kbc_ibbd05_res8_low_yrel5[:, 1],
                   data_kbc_ibbd05_res16_low_yrel5[:, 4], data_kbc_ibbd05_res16_low_yrel5[:, 1],
                   data_kbc_ibbd05_res50_low_yrel5[:, 4], data_kbc_ibbd05_res50_low_yrel5[:, 1],
                   data_kbc_ibbd05_res100_low_yrel5[:, 4], data_kbc_ibbd05_res100_low_yrel5[:, 1],
                   data_kbc_ibbd05_res1000_low_yrel5[:, 4], data_kbc_ibbd05_res1000_low_yrel5[:, 1])

# x-axis labels
axs_res[1,0].set_xlabel(r"$x_{LU}$")
axs_res[1,1].set_xlabel(r"$x_{LU}$")

# y-axis labels
axs_res[0,0].set_ylabel(r"$p$")
axs_res[1,0].set_ylabel(r"$u_{x}$")

# y-axis limits
axs_res[0,0].set_ylim([-0.001,0.001])
axs_res[1,0].set_ylim([-0.02,0.02])

# x-axis limits
axs_res[1,0].set_xlim([0,40])
axs_res[1,1].set_xlim([0,40])

axs_res[0,0].set_title(r"Position $y_{LU} = 1$")
axs_res[0,1].set_title(r"Position $y_{LU} = 5$")

axs_res[1,1].legend(labels=["8 LU/PU", "16 LU/PU", "50 LU/PU", "100 LU/PU", "1000 LU/PU"],fontsize=6)

#plt.suptitle(f"p/ux/uy(x,ti), KBC, ibbd05, lowInlet, RES (8,100,1000), t_PU=4165 (30000steps for res8), res8 at i14000 (crash afterwards)")
plt.savefig(output_base_path + "/" + plot_batch_name + "/" + "lowInlet_RES_TAU_IBB_allRES_pux")
plt.close(fig_res_2)