import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# matplotlib.rcParams.update({'font.size': 7})
# matplotlib.rcParams.update({'lines.linewidth': 0.8})
# matplotlib.rcParams.update({'figure.figsize': [3.4876,3.4876*0.66]})  # 3x [3.2,2] / Gesamt 6.202inches textwidth
# #matplotlib.rcParams.update({'figure.figsize': [5,3]})
# matplotlib.rcParams.update({'figure.autolayout': True})
# matplotlib.rcParams.update({'figure.dpi': 300})
matplotlib.style.use('../figure_style_2column_singleplot.mplstyle')
matplotlib.rcParams.update({'lines.linestyle': '--'})
matplotlib.rcParams.update({'lines.markeredgewidth': 1.2})

# data source
folder = "/home/mbille/lettuce/plotting_mp2"  # HBRS
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
plt.setp(lines, ls="", lw=1, marker="+")

plt.xlabel("year")
plt.ylabel("$C_{D}$")
#plt.xlim([0,165])
plt.ylim([0.5,3])
plt.grid()
plt.xticks(np.arange(1920, 2010+10, 20.0))
plt.legend(["Re 20", "Re 40", "Re 80", "Re 100", "Re 200", "Re 300"], loc=(0.1, 0.05))
#plt.title("Literaturwerte des Widerstandsbeiwerts $C_{D}$ über die Jahre, für verschiedene Reynoldszahlen", wrap=True)
plt.savefig(folder+"/plots/Cd_Literature.png")
plt.show()

