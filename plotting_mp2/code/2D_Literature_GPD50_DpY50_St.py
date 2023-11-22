import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# matplotlib.rcParams.update({'font.size': 7}) # font size was 11
# matplotlib.rcParams.update({'lines.linewidth': 0.8})
# #matplotlib.rcParams.update({'figure.figsize': [6,5]})  # 4,3 war vorher / 3x [3.2,2] / Gesamt 6.202inches textwidth
# matplotlib.rcParams.update({'figure.figsize': [3.4876,3.4876*0.65]})
# matplotlib.rcParams.update({'figure.autolayout': True})
# matplotlib.rcParams.update({'figure.dpi': 300})
matplotlib.style.use('../figure_style_2column_singleplot.mplstyle')
matplotlib.rcParams.update({'lines.linestyle': '--'})

# data source
folder = "/home/mbille/lettuce/plotting_mp2"  # HBRS
#folder =   #BONN

name = "2D_Literature_GPD50_DpY50_St"

data_roshko = np.genfromtxt(folder+"/literature/1953_Roshko_01_St_Re.csv", usemask=True, delimiter=";")
data_Norberg = np.genfromtxt(folder+"/literature/2007_Shu_11_St_Re_comparisonToOthers_St_Norberg.csv", usemask=True, delimiter=";")
data_Williamson88 = np.genfromtxt(folder+"/literature/2007_Shu_11_St_Re_comparisonToOthers_St_Williamson88.csv", usemask=True, delimiter=";")
data_Williamson89 = np.genfromtxt(folder+"/literature/2007_Shu_11_St_Re_comparisonToOthers_St_Williamson89.csv", usemask=True, delimiter=";")
data_shu = np.genfromtxt(folder+"/literature/2007_Shu_11_St_Re_comparisonToOthers_St_Shu.csv", usemask=True, delimiter=";")

lines = plt.plot(data_roshko[:,0], data_roshko[:,1])
plt.setp(lines, ls="", marker="+", color="tab:green", label="Roshko 1953 [exp.]")

lines = plt.plot(data_Norberg[:,0], data_Norberg[:,1])
plt.setp(lines, ls="", marker="+", color="tab:purple",label="Norberg 2003 [num.]")

lines = plt.plot(data_Williamson88[:,0], data_Williamson88[:,1])
plt.setp(lines, ls="", marker="+", color="tab:pink", label="Williamson 1988 [num.]")

lines = plt.plot(data_Williamson89[:,0], data_Williamson89[:,1])
plt.setp(lines, ls="", marker="+", color="tab:olive", label="Williamson 1989 [num.]")

lines = plt.plot(data_shu[:,0], data_shu[:,1])
plt.setp(lines, ls="", marker="+", color="k", label="Shu 2007 [num.]")

data_num_literature = np.genfromtxt(folder+"/literature/St_compare.CSV", usemask=True, delimiter=";")
re = data_num_literature[0,:]
re80 = data_num_literature[2:,0]
re100 = data_num_literature[2:,1]
re200 = data_num_literature[2:,2]
x_data = [*([80]*len(re80)),*([100]*len(re100)),*([200]*len(re200))]
y_data = [*re80,*re100,*re200]
lines = plt.plot(x_data,y_data)
plt.setp(lines, ls="", marker="+", color="tab:blue", label="further tabular lit.")

data = np.genfromtxt(folder+"/data/"+name+".csv", delimiter=",")
# PARAMETERS: GPD50, DpY50, T300 (T1000 for Re50), D2Q9

hwbb_bgk = plt.plot(data[0], data[1], ls="--", lw=0.7, marker=".", color="tab:red", label="HWBB BGK")
ibb_bgk = plt.plot(data[0], data[2], ls="--", lw=0.7, marker=".", color="tab:orange", label="IBB BGK")


plt.xlabel("Re")
plt.ylabel("$St$")
plt.xlim([0,400])
plt.ylim([0.1,0.22])
plt.grid()
#plt.title("Widerstandsbeiwert $C_{D}$ in Abhängigkeit der Domänenbreite in Durchmessern (DpY), für Re = 200", wrap=True)

plt.legend(loc='lower right', fontsize="6")
plt.savefig(folder+"/plots/"+name+".png")
plt.show()

