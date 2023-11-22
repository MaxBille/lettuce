import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# matplotlib.rcParams.update({'font.size': 11})
# matplotlib.rcParams.update({'lines.linewidth': 0.8})
# matplotlib.rcParams.update({'figure.figsize': [4,3]})  # 3x [3.2,2] / Gesamt 6.202inches textwidth
# matplotlib.rcParams.update({'figure.autolayout': True})
# matplotlib.rcParams.update({'figure.dpi': 300})
matplotlib.style.use('../figure_style_2column_dualplot.mplstyle')
matplotlib.rcParams.update({'lines.linestyle': '--'})

# data source
folder = "/home/mbille/lettuce/plotting_mp2"  # HBRS
#folder =   #BONN

data = np.genfromtxt(folder+"/data/2D_DpY_Re200_Cd.csv", delimiter=",")
# PARAMETERS: Re200, GPD30, D2Q9
#print(data)

fig, ax1 = plt.subplots()

fwbb_bgk = plt.plot(data[0][np.isfinite(data[1])], data[1][np.isfinite(data[1])], marker=".", color="tab:blue", label="FWBB BGK")
fwbb_kbc = plt.plot(data[0][np.isfinite(data[2])], data[2][np.isfinite(data[2])], marker="x", color="tab:blue", label="FWBB KBC")
hwbb_bgk = plt.plot(data[0][np.isfinite(data[3])], data[3][np.isfinite(data[3])], marker=".", color="tab:orange", label="HWBB BGK")
hwbb_kbc = plt.plot(data[0][np.isfinite(data[4])], data[4][np.isfinite(data[4])], marker="x", color="tab:orange", label="HWBB KBC")
ibb_bgk = plt.plot(data[0][np.isfinite(data[5])], data[5][np.isfinite(data[5])], marker=".", color="tab:green", label="IBB BGK")
ibb_kbc = plt.plot(data[0][np.isfinite(data[6])], data[6][np.isfinite(data[6])], marker="x", color="tab:green", label="IBB KBC")

plt.xlabel("D/Y")
plt.ylabel("$C_{D}$")
plt.xlim([0,205])
#ylim_lock = ax1.set_ylim()
plt.grid()
#plt.title("Widerstandsbeiwert $C_{D}$ in Abhängigkeit der Domänenbreite in Durchmessern (DpY), für Re = 200", wrap=True)

literature = [1.4,1.31,1.19,1.31,1.33,1.172,1.29,1.45,1.26,1.36,1.4087]
# plt.axhline(y=1.4, color="r", ls="-.", lw=0.5, label="literature")
# for lit in literature[1:]:
#     plt.axhline(y=lit, color="r", ls="-.", lw=0.5)
for lit in literature:
    plt.axhline(y=lit, color="r", ls="", marker="", lw=0.5)
ax2 = ax1.twinx()
ax2.set_yticks(literature, labels=[" "]*len(literature))
ax2.set_ylim(ax1.set_ylim())
ax2.tick_params(color='r', direction='in', width=1.2)

#plt.legend()
plt.savefig(folder+"/plots/2D_DpY_Re200_Cd.png")
plt.show()

