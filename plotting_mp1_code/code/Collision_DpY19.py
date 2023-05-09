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

data = np.genfromtxt(folder+"/literature/Collision_DpY19.CSV", delimiter=";")
lines = plt.plot(data[0], data[1], data[0], data[2], data[0], data[3])
plt.setp(lines, ls="--", lw=1, marker=".")

plt.xlabel("GPD")
plt.ylabel("$C_{D}$")
#plt.xlim([0,50])
plt.grid()
#plt.title("Widerstandsbeiwert $C_{D}$ in Abhängigkeit der Auflösung in GPD für verschiedene Kollisionsoperatoren, für Re = 200", wrap=True)
plt.legend(["BGK", "KBC", "Regularized"])

literature = [1.4,1.31,1.19,1.31,1.33,1.172,1.29,1.45,1.26,1.36,1.4087]
for lit in literature:
    plt.axhline(y=lit, color="r", ls="--", lw=0.5)
plt.savefig(folder+"/plots/Collision_DpY19.png")
plt.show()

