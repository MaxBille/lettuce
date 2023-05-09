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

data = np.genfromtxt(folder+"/literature/Cd_compare.CSV", usemask=True, delimiter=";")
re = data[0,:]
mydata = data[1,:]
re20 = data[2:,0]
re40 = data[2:,1]
re80 = data[2:,2]
re100 = data[2:,3]
re200 = data[2:,4]
re300 = data[2:,5]

data_tritton = np.genfromtxt(folder+"/literature/1959_Tritton_01_logCd_logRe.csv", usemask=True, delimiter=";")
data_relf = np.genfromtxt(folder+"/literature/1959_Tritton_02_logCd_logRe_comparisonToOthers_Relf.csv", usemask=True, delimiter=";")
data_weiselsberger = np.genfromtxt(folder+"/literature/1959_Tritton_02_logCd_logRe_comparisonToOthers_Weiselsberger.csv", usemask=True, delimiter=";")
data_hoerner = np.genfromtxt(folder+"/literature/Hoerner_S52_DragCircularCylinder.csv", usemask=True, delimiter=";")

lines = plt.plot(data_tritton[:,0], data_tritton[:,1])
plt.setp(lines, ls="", lw=1, marker="+", color="tab:green")

lines = plt.plot(data_relf[:,0], data_relf[:,1])
plt.setp(lines, ls="", lw=1, marker="+", color="tab:purple")

lines = plt.plot(data_weiselsberger[:,0], data_weiselsberger[:,1])
plt.setp(lines, ls="", lw=1, marker="+", color="tab:orange")

lines = plt.plot(data_hoerner[:,0], data_hoerner[:,1])
plt.setp(lines, ls="", lw=1, marker="+", color="k")

lines = plt.plot([20]*len(re20),re20,[40]*len(re40),re40,[80]*len(re80),re80,[100]*len(re100),re100,[200]*len(re200),re200,[300]*len(re300),re300)
plt.setp(lines, ls="", lw=1, marker="+", color="tab:blue")

lines = plt.plot(re,mydata)
plt.setp(lines, ls="--", lw=1.2, marker=".", markersize=8,color="tab:red")

#print(re)
#print(mydata)

plt.xlabel("Re")
plt.ylabel("$C_{D}$")
plt.xlim([0,400])
plt.ylim([0.5,3])
plt.grid()
#plt.xticks(np.arange(1990, 2010+5, 5.0))
plt.legend(["Tritton 1959 [exp.]", "Relf 1913 [exp.]", "Weiselsberger 1922 [exp.]", "Hoerner 1965 [mixed]", "sonstig. num. Literatur"])
#plt.title("Widerstandsbeiwert $C_{D}$ für verschiedene Reynoldszahlen, \nVergleich mit Literatur", wrap=True)

plt.savefig(folder+"/plots/Cd_compare_extended.png")
plt.show()

