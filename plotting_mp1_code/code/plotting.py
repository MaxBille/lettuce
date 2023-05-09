import matplotlib.pyplot as plt
import numpy as np

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
lines = plt.plot([20]*len(re20),re20,[40]*len(re40),re40,[80]*len(re80),re80,[100]*len(re100),re100,[200]*len(re200),re200,[300]*len(re300),re300)
plt.setp(lines, ls="", lw=1, marker="+", color="tab:blue")
lines = plt.plot(re,mydata)
plt.setp(lines, ls="--", lw=1, marker=".", color="tab:orange")

print(re)
print(mydata)

plt.xlabel("Re")
plt.ylabel("$C_{D}$")
#plt.xlim([0,165])
plt.ylim([0.5,3])
plt.grid()
#plt.xticks(np.arange(1990, 2010+5, 5.0))
plt.legend(["Literatur"])
plt.title("Widerstandsbeiwert $C_{D}$ für verschiedene Reynoldszahlen, \nVergleich mit Literatur", wrap=True)

plt.savefig("/home/max/Desktop/plots/Cd_compare.png")
plt.show()

