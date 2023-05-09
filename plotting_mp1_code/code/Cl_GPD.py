import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({'font.size': 11})
matplotlib.rcParams.update({'lines.linewidth': 0.8})
matplotlib.rcParams.update({'figure.figsize': [5,4]})  # 3x [3.2,2] / Gesamt 6.202inches textwidth
matplotlib.rcParams.update({'figure.autolayout': True})

# data source
folder = "/home/mbille/lettuce/plotting_mp1_code"  # HBRS
#folder =   #BONN

data = np.genfromtxt(folder+"/literature/Cl_GPD.CSV", delimiter=";")
lines = plt.plot(*data)
plt.setp(lines, ls="--", lw=1, marker=".")

plt.xlabel("GPD")
plt.ylabel("$C_{L}$")
plt.xlim([0,165])
plt.grid()
#plt.title("Mittlerer Auftriebsbeiwert $C_{L}$ in Abhängigkeit der Auflösung in Gitterpunkten pro Durchmesser (GPD), für Re = 200", wrap=True)

literature = [0.75,0.65,0.69,0.68,0.67,0.5,0.63,0.47,0.64,0.64,0.725]
for lit in literature:
    plt.axhline(y=lit, color="r", ls="--", lw=0.5)
plt.savefig(folder+"/plots/Cl_GPD.png")
plt.show()

