import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# matplotlib.rcParams.update({'font.size': 7}) # font size was 11
# matplotlib.rcParams.update({'lines.linewidth': 0.8})
# #matplotlib.rcParams.update({'figure.figsize': [6,3]})  # 4,3 war vorher / 3x [3.2,2] / Gesamt 6.202inches textwidth
# matplotlib.rcParams.update({'figure.figsize': [3.4876,3.4876*0.65]})
# matplotlib.rcParams.update({'figure.autolayout': True})
# matplotlib.rcParams.update({'figure.dpi': 300})
matplotlib.style.use('../figure_style_2column_singleplot.mplstyle')
matplotlib.rcParams.update({'lines.linestyle': '--'})

# data source
folder = "/home/mbille/lettuce/plotting_mp2"  # HBRS
#folder =   #BONN

name = "2D_Literature_GPD50_DpY50_Cl"

data = np.genfromtxt(folder+"/data/"+name+".csv", delimiter=",")
# PARAMETERS: GPD50, DpY50, T300 (T1000 for Re50), D2Q9

hwbb_bgk = plt.plot(data[0], data[1], marker=".", color="tab:red", label="HWBB BGK")
ibb_bgk = plt.plot(data[0], data[2], marker=".", color="tab:orange", label="IBB BGK")

data_num_literature = np.genfromtxt(folder+"/literature/Cl_compare.CSV", usemask=True, delimiter=";")
re = data_num_literature[0,:]
re80 = data_num_literature[2:,0]
re100 = data_num_literature[2:,1]
re200 = data_num_literature[2:,2]
x_data = [*([80]*len(re80)),*([100]*len(re100)),*([200]*len(re200))]
y_data = [*re80,*re100,*re200]
lines = plt.plot(x_data,y_data)
plt.setp(lines, ls="", lw=1, marker="+", color="tab:blue", label="tabular lit.")

plt.xlabel("Re")
plt.ylabel("$C_{L}$")
#plt.xlim([0,301])
plt.ylim([0,1])
plt.grid()
#plt.title("Widerstandsbeiwert $C_{D}$ in Abhängigkeit der Domänenbreite in Durchmessern (DpY), für Re = 200", wrap=True)

plt.legend()
plt.savefig(folder+"/plots/"+name+".png")
plt.show()

