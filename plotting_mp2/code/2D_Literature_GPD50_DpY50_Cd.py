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

name = "2D_Literature_GPD50_DpY50_Cd"

data = np.genfromtxt(folder+"/data/"+name+".csv", delimiter=",")
# PARAMETERS: GPD50, DpY50, T300 (T1000 for Re50), D2Q9

hwbb_bgk = plt.plot(data[0], data[1], marker=".", color="tab:red", label="HWBB BGK")
ibb_bgk = plt.plot(data[0], data[2], marker=".", color="tab:orange", label="IBB BGK")

data_tritton = np.genfromtxt(folder+"/literature/1959_Tritton_01_logCd_logRe.csv", usemask=True, delimiter=";")
data_relf = np.genfromtxt(folder+"/literature/1959_Tritton_02_logCd_logRe_comparisonToOthers_Relf.csv", usemask=True, delimiter=";")
data_weiselsberger = np.genfromtxt(folder+"/literature/1959_Tritton_02_logCd_logRe_comparisonToOthers_Weiselsberger.csv", usemask=True, delimiter=";")
data_hoerner = np.genfromtxt(folder+"/literature/Hoerner_S52_DragCircularCylinder.csv", usemask=True, delimiter=";")

lines_tritton = plt.plot(data_tritton[:,0], data_tritton[:,1])
plt.setp(lines_tritton, ls="", marker="+", color="tab:green", label="Tritton 1959 [exp.]")

lines_relf = plt.plot(data_relf[:,0], data_relf[:,1])
plt.setp(lines_relf, ls="", marker="+", color="tab:purple", label="Relf 1913 [exp.]")

lines_weiselsberger = plt.plot(data_weiselsberger[:,0], data_weiselsberger[:,1])
plt.setp(lines_weiselsberger, ls="", marker="+", color="tab:brown", label="Weiselsberger 1922 [exp.]")

lines_hoerner = plt.plot(data_hoerner[:,0], data_hoerner[:,1])
plt.setp(lines_hoerner, ls="", marker="+", color="k", label="Hoerner 1965 [mixed]")

data_num_literature = np.genfromtxt(folder+"/literature/Cd_compare.CSV", usemask=True, delimiter=";")
re = data_num_literature[0,:]
re20 = data_num_literature[2:,0]
re40 = data_num_literature[2:,1]
re80 = data_num_literature[2:,2]
re100 = data_num_literature[2:,3]
re200 = data_num_literature[2:,4]
re300 = data_num_literature[2:,5]

# create single data-stream:
x_data = [*([20]*len(re20)),*([40]*len(re40)),*([80]*len(re80)),*([100]*len(re100)),*([200]*len(re200)),*([300]*len(re300))]
y_data = [*re20,*re40,*re80,*re100,*re200,*re300]
#lines = plt.plot([20]*len(re20),re20,[40]*len(re40),re40,[80]*len(re80),re80,[100]*len(re100),re100,[200]*len(re200),re200,[300]*len(re300),re300)
lines = plt.plot(x_data,y_data)
plt.setp(lines, ls="", marker="+", color="tab:blue", label="further tabular lit.")

plt.xlabel("Re")
plt.ylabel("$C_{D}$")
plt.xlim([0,305])
plt.ylim([0.75,3])
plt.grid()
#plt.title("Widerstandsbeiwert $C_{D}$ in Abhängigkeit der Domänenbreite in Durchmessern (DpY), für Re = 200", wrap=True)

plt.legend()
plt.savefig(folder+"/plots/"+name+".png")
plt.show()

