import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# for tick-str formatting
from matplotlib.ticker import FormatStrFormatter

# matplotlib.rcParams.update({'font.size': 7}) # font size was 11
# matplotlib.rcParams.update({'lines.linewidth': 0.8})
# #matplotlib.rcParams.update({'figure.figsize': [6,3]})  # 4,3 war vorher / 3x [3.2,2] / Gesamt 6.202inches textwidth
# matplotlib.rcParams.update({'figure.figsize': [3.4876,3.4876*0.65]})
# matplotlib.rcParams.update({'figure.autolayout': True})
# matplotlib.rcParams.update({'figure.dpi': 300})
matplotlib.style.use('../figure_style_2column_singleplot.mplstyle')
matplotlib.rcParams.update({'lines.linestyle': '--'})
#matplotlib.rcParams.update({'font.size': 8}) # font size was 11

# data source
folder = "/home/mbille/lettuce/plotting_mp2"  # HBRS
#folder =   #BONN

name = "2D3D_compare_Cl"
name2d = "2D3D_2D_Cl"
name3d = "2D3D_3D_Cl"

data2d = np.genfromtxt(folder+"/data/"+name2d+".csv", delimiter=",")
data3d = np.genfromtxt(folder+"/data/"+name3d+".csv", delimiter=",")
# PARAMETERS: GPD50, DpY50, T300 (T1000 for Re50), D2Q9

fig, ax1 = plt.subplots()

hwbb_bgk_2d = plt.plot(data2d[0], data2d[1], marker=".", color="tab:red", label="2D HWBB")
ibb_bgk_2d = plt.plot(data2d[0], data2d[2], marker=".", color="tab:orange", label="2D IBB")

hwbb_bgk_3d = plt.plot(data3d[0], data3d[1], marker="x", color="tab:red", label="3D HWBB")
ibb_bgk_3d = plt.plot(data3d[0], data3d[2], marker="x", color="tab:orange", label="3D IBB")

show_literature=False
if show_literature:
    data_num_literature = np.genfromtxt(folder + "/literature/Cl_compare.CSV", usemask=True, delimiter=";")
    re = data_num_literature[0, :]
    re80 = data_num_literature[2:, 0]
    re100 = data_num_literature[2:, 1]
    re200 = data_num_literature[2:, 2]
    x_data = [*([80] * len(re80)), *([100] * len(re100)), *([200] * len(re200))]
    y_data = [*re80, *re100, *re200]
    lines = plt.plot(x_data, y_data)
    plt.setp(lines, ls="", marker="+", color="tab:blue", label="num. Literatur")

plt.xlabel("Re")
plt.ylabel("$C_{L}$")
plt.xlim([99,301])
plt.ylim([0.2,1.1])
plt.grid()
#plt.title("Widerstandsbeiwert $C_{D}$ in Abhängigkeit der Domänenbreite in Durchmessern (DpY), für Re = 200", wrap=True)

ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f  '))

plt.legend()
plt.savefig(folder+"/plots/"+name+".png")
plt.show()

