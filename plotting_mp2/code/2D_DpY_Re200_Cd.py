import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({'font.size': 11})
matplotlib.rcParams.update({'lines.linewidth': 0.8})
matplotlib.rcParams.update({'figure.figsize': [4,3]})  # 3x [3.2,2] / Gesamt 6.202inches textwidth
matplotlib.rcParams.update({'figure.autolayout': True})

# data source
folder = "/home/mbille/lettuce/plotting_mp2"  # HBRS
#folder =   #BONN

data = np.genfromtxt(folder+"/data/2D_DpY_Re200_Cd.csv", delimiter=",")
# PARAMETERS: Re200, GPD30, D2Q9

fwbb_bgk = plt.plot(data[0], data[1], ls="--", lw=1, marker=".", color="tab:blue", label="FWBB BGK")
fwbb_kbc = plt.plot(data[0], data[2], ls="--", lw=1, marker="x", color="tab:blue", label="FWBB KBC")
hwbb_bgk = plt.plot(data[0], data[3], ls="--", lw=1, marker=".", color="tab:orange", label="HWBB BGK")
hwbb_kbc = plt.plot(data[0], data[4], ls="--", lw=1, marker="x", color="tab:orange", label="HWBB KBC")
ibb_bgk = plt.plot(data[0], data[5], ls="--", lw=1, marker=".", color="tab:green", label="IBB BGK")
ibb_kbc = plt.plot(data[0], data[6], ls="--", lw=1, marker="x", color="tab:green", label="IBB KBC")

plt.xlabel("D/Y")
plt.ylabel("$C_{D}$")
plt.xlim([0,205])
plt.grid()
#plt.title("Widerstandsbeiwert $C_{D}$ in Abhängigkeit der Domänenbreite in Durchmessern (DpY), für Re = 200", wrap=True)

literature = [1.4,1.31,1.19,1.31,1.33,1.172,1.29,1.45,1.26,1.36,1.4087]
plt.axhline(y=1.4, color="r", ls="-.", lw=0.5, label="literature")
for lit in literature[1:]:
    plt.axhline(y=lit, color="r", ls="-.", lw=0.5)
#plt.legend()
plt.savefig(folder+"/plots/2D_DpY_Re200_Cd.png")
plt.show()

