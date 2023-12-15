import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#matplotlib.rcParams.update({'font.size': 11})
#matplotlib.rcParams.update({'lines.linewidth': 0.8})
#matplotlib.rcParams.update({'figure.figsize': [4,3]})  # 3x [3.2,2] / Gesamt 6.202inches textwidth
matplotlib.style.use('../figure_style_2column_dualplot.mplstyle')
matplotlib.rcParams.update({'lines.linestyle': '--'})
#matplotlib.rcParams.update({'figure.figsize': [3.4876/2,3.4876/2]})

# data source
folder = "/home/mbille/lettuce/plotting_mp2"  # HBRS
#folder =   #BONN

data = np.genfromtxt(folder+"/data/2D_DpY_Re40_Cd.csv", delimiter=",")
# PARAMETERS: Re40, GPD30, D2Q9
#lines_bgk = plt.plot(data[0], data[1], data[0], data[3], data[0], data[5])
#lines_kbc = plt.plot(data[0], data[2], data[0], data[4], data[0], data[6])

fig, ax1 = plt.subplots()

# fwbb_bgk = plt.plot(data[0], data[1], marker=".", color="tab:blue", label="FWBB BGK")
# fwbb_kbc = plt.plot(data[0], data[2], marker="x", color="tab:blue", label="FWBB KBC")
# hwbb_bgk = plt.plot(data[0], data[3], marker=".", color="tab:orange", label="HWBB BGK")
# hwbb_kbc = plt.plot(data[0], data[4], marker="x", color="tab:orange", label="HWBB KBC")
# ibb_bgk = plt.plot(data[0], data[5], marker=".", color="tab:green", label="IBB BGK")
# ibb_kbc = plt.plot(data[0], data[6], marker="x", color="tab:green", label="IBB KBC")
fwbb_bgk = plt.plot(data[0][np.isfinite(data[1])], data[1][np.isfinite(data[1])], marker=".", color="tab:blue", label="FWBB BGK")
fwbb_kbc = plt.plot(data[0][np.isfinite(data[2])], data[2][np.isfinite(data[2])], marker="x", color="tab:blue", label="FWBB KBC")
hwbb_bgk = plt.plot(data[0][np.isfinite(data[3])], data[3][np.isfinite(data[3])], marker=".", color="tab:orange", label="HWBB BGK")
hwbb_kbc = plt.plot(data[0][np.isfinite(data[4])], data[4][np.isfinite(data[4])], marker="x", color="tab:orange", label="HWBB KBC")
ibb_bgk = plt.plot(data[0][np.isfinite(data[5])], data[5][np.isfinite(data[5])], marker=".", color="tab:green", label="IBB BGK")
ibb_kbc = plt.plot(data[0][np.isfinite(data[6])], data[6][np.isfinite(data[6])], marker="x", color="tab:green", label="IBB KBC")
#plt.setp(lines_bgk, marker=".")
#plt.setp(lines_kbc, marker="x")

plt.xlabel("D/Y")
plt.ylabel("$C_{D}$")
plt.xlim([0, 205])
plt.grid()
#plt.title("Widerstandsbeiwert $C_{D}$ in Abhängigkeit der Domänenbreite in Durchmessern (DpY), für Re = 40", wrap=True)

literature = [1.7,1.48,1.522,1.498,1.52,1.62,1.6,1.63,1.52,1.55,1.52,1.62]
# plt.axhline(y=1.7, color="r", ls="-.", lw=0.5, label="literature")
# for lit in literature[1:]:
#     plt.axhline(y=lit, color="r", ls="-.", lw=0.5)
for lit in literature:
    plt.axhline(y=lit, color="r", ls="", marker="", lw=0.5)

ylim_lock = ax1.set_ylim()
plt.axhline(y=100, color="r", ls="", marker="_",lw=0.5, label="literature")

ax2 = ax1.twinx()
ax2.set_yticks(literature, labels=[" "]*len(literature))
ax2.set_ylim(ylim_lock)
ax1.set_ylim(ylim_lock)
ax2.tick_params(color='r', direction='in', width=1.2)

ax1.legend()

plt.savefig(folder+"/plots/2D_DpY_Re40_Cd.png")
plt.show()

# handles, labels = ax1.get_legend_handles_labels()
# order = np.arange(len(handles))
# ax1.legend([handles[idx] for idx in [0,1,2]], [labels[idx] for idx in [0,1,2]],
#            loc=3, bbox_to_anchor=(0, 0.82), frameon=True, ncol=3, edgecolor="white", facecolor="white", columnspacing=1, fontsize=8)