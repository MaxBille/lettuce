import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# matplotlib.rcParams.update({'font.size': 7})
# matplotlib.rcParams.update({'lines.linewidth': 0.8})
# #matplotlib.rcParams.update({'figure.figsize': [6,4]})  # 3x [3.2,2] / Gesamt 6.202inches textwidth
# matplotlib.rcParams.update({'figure.figsize': [3.4876,3.4876*0.66]})
# matplotlib.rcParams.update({'figure.autolayout': True})
# matplotlib.rcParams.update({'figure.dpi': 300})
matplotlib.style.use('../figure_style_2column_singleplot.mplstyle')
matplotlib.rcParams.update({'lines.linestyle': '--'})
matplotlib.rcParams.update({'lines.markeredgewidth': 1.2})

# data source
folder = "/home/mbille/lettuce/plotting_mp2"  # HBRS
#folder =   #BONN

data = np.genfromtxt(folder+"/literature/St_Literatur.CSV", usemask=True, delimiter=";")
years = data[:,0]
re80 = data[:,1]
re100 = data[:,2]
re200 = data[:,3]
print(years)
lines = plt.plot(years,re80,years,re100,years,re200)#,years,re300)
plt.setp(lines, ls="", lw=1, marker="+")

plt.xlabel("year")
plt.ylabel("$St$")
#plt.xlim([0,165])
#plt.ylim([0.1,0.25])
plt.grid()
plt.xticks(np.arange(1990, 2010+5, 5.0))
plt.yticks(np.arange(0.1,0.26,0.05))
plt.legend(["Re 80", "Re 100", "Re 200", "Re 300"])
#plt.title("Literaturwerte der Strouhal-Zahl $St$ über die Jahre, für verschiedene Reynoldszahlen", wrap=True)
plt.savefig(folder+"/plots/St_Literature.png")
plt.show()

