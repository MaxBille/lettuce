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

data = np.genfromtxt(folder+"/literature/St_GPD.CSV", delimiter=";")
lines = plt.plot(*data)
plt.setp(lines, ls="--", lw=1, marker=".")

plt.xlabel("GPD")
plt.ylabel("$St$")
plt.xlim([0,165])
plt.grid()
#plt.title("Strouhal-Zahl $St$ in Abhängigkeit der Auflösung in Gitterpunkten pro Durchmesser (GPD), für Re = 200", wrap=True)

literature = [0.2,0.192,0.196,0.202,0.195,0.201,0.192,0.191]
for lit in literature:
    plt.axhline(y=lit, color="r", ls="--", lw=0.5)
plt.savefig(folder+"/plots/St_GPD.png")
plt.show()

