import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({'font.size': 11})
matplotlib.rcParams.update({'lines.linewidth': 0.8})
matplotlib.rcParams.update({'figure.figsize': [4,3]})  # 3x [3.2,2] / Gesamt 6.202inches textwidth
matplotlib.rcParams.update({'figure.autolayout': True})

# data source
folder = "/home/mbille/lettuce/plotting_mp1_code"  # HBRS
#folder =   #BONN

data = np.genfromtxt(folder+"/literature/Cd_DpY_Re200.CSV", delimiter=";")
lines = plt.plot(*data)
plt.setp(lines, ls="--", lw=1, marker=".")

plt.xlabel("DpY")
plt.ylabel("$C_{D}$")
plt.xlim([0,80])
plt.grid()
#plt.title("Widerstandsbeiwert $C_{D}$ in Abhängigkeit der Domänenbreite in Durchmessern (DpY), für Re = 200", wrap=True)

literature = [1.4,1.31,1.19,1.31,1.33,1.172,1.29,1.45,1.26,1.36,1.4087]
for lit in literature:
    plt.axhline(y=lit, color="r", ls="--", lw=0.5)
plt.savefig(folder+"/plots/Cd_DpY_Re200.png")
plt.show()

