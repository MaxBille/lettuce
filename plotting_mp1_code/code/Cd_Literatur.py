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

data = np.genfromtxt(folder+"/literature/Cd_Literatur.CSV", usemask=True, delimiter=";")
years = data[:,0]
re20 = data[:,1]
re40 = data[:,2]
re80 = data[:,3]
re100 = data[:,4]
re200 = data[:,5]
#re300 = data[:,6]

#print(data[0,0])
lines = plt.plot(years,re20,years,re40,years,re80,years,re100,years,re200)#,years,re300)
plt.setp(lines, ls="", lw=1, marker="+", markersize=6, markeredgewidth=2)

plt.xlabel("Jahr")
plt.ylabel("$C_{D}$")
#plt.xlim([0,165])
plt.ylim([0.5,3])
plt.grid()
plt.xticks(np.arange(1920, 2010+10, 10.0))
plt.legend(["Re 20", "Re 40", "Re 80", "Re 100", "Re 200", "Re 300"])
#plt.title("Literaturwerte des Widerstandsbeiwerts $C_{D}$ über die Jahre, für verschiedene Reynoldszahlen", wrap=True)
plt.savefig(folder+"/plots/Cd_Literatur.png")
plt.show()

