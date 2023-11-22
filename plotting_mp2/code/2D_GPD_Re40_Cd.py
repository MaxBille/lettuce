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

name = "2D_GPD_Re40_Cd"

data = np.genfromtxt(folder+"/data/"+name+".csv", delimiter=",")
# PARAMETERS: Re40, DpY19, D2Q9

fig, ax1 = plt.subplots()

fwbb_bgk = plt.plot(data[0], data[1], marker=".", color="tab:blue", label="FWBB BGK")
#fwbb_reg = plt.plot(data[0], data[2], marker="^", color="tab:blue", label="FWBB REG")
fwbb_kbc = plt.plot(data[0], data[3], marker="x", color="tab:blue", label="FWBB KBC")

hwbb_kbc = plt.plot(data[0], data[4], marker="x", color="tab:orange", label="HWBB KBC")

ibb_bgk = plt.plot(data[0], data[5], marker=".", color="tab:green", label="IBB BGK")
#ibb_reg = plt.plot(data[0], data[6], marker="^", color="tab:green", label="IBB REG")
ibb_kbc = plt.plot(data[0], data[7], marker="x", color="tab:green", label="IBB KBC")

plt.xlabel("GPD")
plt.ylabel("$C_{D}$")
plt.xlim([9,61])
#plt.ylim([0.18,0.21])
plt.grid()
#plt.title("Widerstandsbeiwert $C_{D}$ in Abhängigkeit der Domänenbreite in Durchmessern (DpY), für Re = 200", wrap=True)

# converged values
plt.axhline(y=data[1,-1], color='tab:blue', ls=":", lw=0.7, label="FWBB BGK GPD100")
plt.axhline(y=data[5,-1], color='tab:green', ls=":", lw=0.7, label="IBB BGK GPD100")

literature = [1.7,1.48,1.522,1.498,1.52,1.62,1.6,1.63,1.52,1.55,1.52,1.62]
# plt.axhline(y=1.7, color="r", ls="-.", lw=0.5, label="literature")
# for lit in literature[1:]:
#     plt.axhline(y=lit, color="r", ls="-.", lw=0.5)
for lit in literature:
    plt.axhline(y=lit, color="r", ls="", marker="", lw=0.5)
ax2 = ax1.twinx()
ax2.set_yticks(literature, labels=[" "]*len(literature))
ax2.set_ylim(ax1.set_ylim())
ax2.tick_params(color='r', direction='in', width=1.2)
#ax1.legend(fontsize="6")

plt.savefig(folder+"/plots/"+name+".png")
plt.show()

