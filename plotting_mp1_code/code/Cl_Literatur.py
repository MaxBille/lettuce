import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({'font.size': 11})
matplotlib.rcParams.update({'lines.linewidth': 0.8})
matplotlib.rcParams.update({'figure.figsize': [6,4]})  # 3x [3.2,2] / Gesamt 6.202inches textwidth
matplotlib.rcParams.update({'figure.autolayout': True})

# data source
folder = "/home/mbille/lettuce/plotting_mp1_code"  # HBRS
#folder =   #BONN

data = np.genfromtxt(folder+"/literature/Cl_Literatur.CSV", usemask=True, delimiter=";")
years = data[:,0]
re100 = data[:,1]
re200 = data[:,2]
lines = plt.plot(years,re100,years,re200)#,years,re300)
plt.setp(lines, ls="", lw=1, marker="+", markersize=6, markeredgewidth=2)
print(years)
plt.xlabel("Jahr")
plt.ylabel("$C_{L}$")
#plt.xlim([0,165])
plt.ylim([0,1])
plt.grid()
plt.xticks(np.arange(1985, 2010+5, 5.0))
plt.legend(["Re 100", "Re 200"])
#plt.title("Literaturwerte des Auftriebsbeiwerts $C_{L}$ über die Jahre, für verschiedene Reynoldszahlen", wrap=True)
plt.savefig(folder+"/plots/Cl_Literatur.png")
plt.show()

