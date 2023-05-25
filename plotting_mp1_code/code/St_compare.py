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

data = np.genfromtxt(folder+"/literature/St_compare.CSV", usemask=True, delimiter=";")
re = data[0,:]
mydata = data[1,:]

data_roshko = np.genfromtxt(folder+"/literature/1953_Roshko_01_St_Re.csv", usemask=True, delimiter=";")
data_Norberg = np.genfromtxt(folder+"/literature/2007_Shu_11_St_Re_comparisonToOthers_St_Norberg.csv", usemask=True, delimiter=";")
data_Williamson88 = np.genfromtxt(folder+"/literature/2007_Shu_11_St_Re_comparisonToOthers_St_Williamson88.csv", usemask=True, delimiter=";")
data_Williamson89 = np.genfromtxt(folder+"/literature/2007_Shu_11_St_Re_comparisonToOthers_St_Williamson89.csv", usemask=True, delimiter=";")
data_shu = np.genfromtxt(folder+"/literature/2007_Shu_11_St_Re_comparisonToOthers_St_Shu.csv", usemask=True, delimiter=";")

lines = plt.plot(data_roshko[:,0], data_roshko[:,1])
plt.setp(lines, ls="", lw=1, marker="+", color="tab:green", label="Roshko 1953 [exp.]")

lines = plt.plot(data_Norberg[:,0], data_Norberg[:,1])
plt.setp(lines, ls="", lw=1, marker="+", color="tab:purple",label="Norberg 2003 [num.]")

lines = plt.plot(data_Williamson88[:,0], data_Williamson88[:,1])
plt.setp(lines, ls="", lw=1, marker="+", color="tab:orange", label="Williamson 1988 [num.]")

lines = plt.plot(data_Williamson89[:,0], data_Williamson89[:,1])
plt.setp(lines, ls="", lw=1, marker="+", color="tab:olive", label="Williamson 1989 [num.]")

lines = plt.plot(data_shu[:,0], data_shu[:,1])
plt.setp(lines, ls="", lw=1, marker="+", color="k", label="Shu 2007 [num.]")

#re20 = data[2:,0]
#re40 = data[2:,1]
re80 = data[2:,0]
re100 = data[2:,1]
re200 = data[2:,2]
#re300 = data[2:,5]
# lines = plt.plot(#[20]*len(re20),re20,
#                  #[40]*len(re40),re40,
#                  [80]*len(re80),re80,
#                  [100]*len(re100),re100,
#                  [200]*len(re200),re200,
#                  #[300]*len(re300),re300
#                  )
# plt.setp(lines, ls="", lw=1, marker="+", color="tab:blue")
#
# lines = plt.plot(re,mydata)
# plt.setp(lines, ls="--", lw=1.2, marker=".", markersize=8,color="tab:red")

# create single data-stream:
x_data = [*([80]*len(re80)),*([100]*len(re100)),*([200]*len(re200))]
y_data = [*re80,*re100,*re200]
lines = plt.plot(x_data,y_data)
plt.setp(lines, ls="", lw=1, marker="+", color="tab:blue", label="sonstig. num. Literatur")

mylines = plt.plot(re,mydata)
plt.setp(mylines, ls="--", lw=1.2, marker=".", markersize=8,color="tab:red", label="Lettuce")

plt.xlabel("Re")
plt.ylabel("$St$")
plt.xlim([0,400])
plt.ylim([0,0.3])
plt.grid()
#plt.xticks(np.arange(1990, 2010+5, 5.0))
#plt.legend(["Roshko 1953 [exp.]","Norberg 2003 [num.]","Williamson 1988 [num.]", "Williamson 1989 [num.]", "Shu 2007 [num.]","sonstig. num. Literatur"])
plt.legend()
#plt.title("Strouhal-Zahl $St$ für verschiedene Reynoldszahlen, \nVergleich mit Literatur", wrap=True)

plt.savefig(folder+"/plots/St_compare_extended.png")
plt.show()

