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

data = np.genfromtxt(folder+"/data/2D_DpY_Re40_Cd.csv", delimiter=",")
# PARAMETERS: Re40, GPD30, D2Q9
#lines_bgk = plt.plot(data[0], data[1], data[0], data[3], data[0], data[5])
#lines_kbc = plt.plot(data[0], data[2], data[0], data[4], data[0], data[6])

fwbb_bgk = plt.plot(data[0], data[1], ls="--", lw=1, marker=".", color="tab:blue", label="FWBB BGK")
fwbb_kbc = plt.plot(data[0], data[2], ls="--", lw=1, marker="x", color="tab:blue", label="FWBB KBC")
hwbb_bgk = plt.plot(data[0], data[3], ls="--", lw=1, marker=".", color="tab:orange", label="HWBB BGK")
hwbb_kbc = plt.plot(data[0], data[4], ls="--", lw=1, marker="x", color="tab:orange", label="HWBB KBC")
ibb_bgk = plt.plot(data[0], data[5], ls="--", lw=1, marker=".", color="tab:green", label="IBB BGK")
ibb_kbc = plt.plot(data[0], data[6], ls="--", lw=1, marker="x", color="tab:green", label="IBB KBC")
#plt.setp(lines_bgk, ls="--", lw=1, marker=".")
#plt.setp(lines_kbc, ls="--", lw=1, marker="x")

plt.xlabel("D/Y")
plt.ylabel("$C_{D}$")
plt.xlim([0, 205])
plt.grid()
#plt.title("Widerstandsbeiwert $C_{D}$ in Abhängigkeit der Domänenbreite in Durchmessern (DpY), für Re = 40", wrap=True)

literature = [1.7,1.48,1.522,1.498,1.52,1.62,1.6,1.63,1.52,1.55,1.52,1.62]
plt.axhline(y=1.7, color="r", ls="-.", lw=0.5, label="literature")
for lit in literature[1:]:
    plt.axhline(y=lit, color="r", ls="-.", lw=0.5)
plt.legend()

plt.savefig(folder+"/plots/2D_DpY_Re40_Cd.png")
plt.show()

