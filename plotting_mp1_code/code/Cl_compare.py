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

data = np.genfromtxt(folder+"/literature/Cl_compare.CSV", usemask=True, delimiter=";")
re = data[0,:]
mydata = data[1,:]
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
# lines = plt.plot(re,mydata)
# plt.setp(lines, ls="--", lw=1.2, marker=".", markersize=8, color="tab:orange")

# create single data-stream:
x_data = [*([80]*len(re80)),*([100]*len(re100)),*([200]*len(re200))]
y_data = [*re80,*re100,*re200]
lines = plt.plot(x_data,y_data)
plt.setp(lines, ls="", lw=1, marker="+", color="tab:blue", label="num. Literatur")

mylines = plt.plot(re,mydata)
plt.setp(mylines, ls="--", lw=1.2, marker=".", markersize=8,color="tab:red", label="Lettuce")

plt.xlabel("Re")
plt.ylabel("$C_{L}$")
#plt.xlim([0,165])
plt.ylim([0,1])
plt.grid()
#plt.xticks(np.arange(1990, 2010+5, 5.0))
plt.legend()
#plt.title("Auftriebsbeiwert $C_{L}$ für verschiedene Reynoldszahlen, \nVergleich mit Literatur", wrap=True)

plt.savefig(folder+"/plots/Cl_compare.png")
plt.show()