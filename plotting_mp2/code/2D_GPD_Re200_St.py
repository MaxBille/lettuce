import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# matplotlib.rcParams.update({'font.size': 9}) # font size was 11
# matplotlib.rcParams.update({'lines.linewidth': 0.8})
# matplotlib.rcParams.update({'figure.figsize': [6,3]})  # 4,3 war vorher / 3x [3.2,2] / Gesamt 6.202inches textwidth
# matplotlib.rcParams.update({'figure.autolayout': True})
matplotlib.style.use('../figure_style_2column_singleplot.mplstyle')
matplotlib.rcParams.update({'lines.linestyle': '--'})

# data source
folder = "/home/mbille/lettuce/plotting_mp2"  # HBRS
#folder =   #BONN

name = "2D_GPD_Re200_St"

data = np.genfromtxt(folder+"/data/"+name+".csv", delimiter=",")
# PARAMETERS: Re200, DpY19, D2Q9

fig, ax1 = plt.subplots()

fwbb_bgk = plt.plot(data[0], data[1], marker=".", color="tab:blue", label="FWBB BGK")
#fwbb_reg = plt.plot(data[0], data[2], marker="^", color="tab:blue", label="FWBB REG")
fwbb_kbc = plt.plot(data[0], data[3], marker="x", color="tab:blue", label="FWBB KBC")

hwbb_bgk = plt.plot(data[0], data[4], marker=".", color="tab:orange", label="HWBB BGK")
#hwbb_reg = plt.plot(data[0], data[5], marker="^", color="tab:orange", label="HWBB REG")
hwbb_kbc = plt.plot(data[0], data[6], marker="x", color="tab:orange", label="HWBB KBC")

ibb_bgk = plt.plot(data[0], data[7], marker=".", color="tab:green", label="IBB BGK")
#ibb_reg = plt.plot(data[0], data[8], marker="^", color="tab:green", label="IBB REG")
ibb_kbc = plt.plot(data[0], data[9], marker="x", color="tab:green", label="IBB KBC")

plt.xlabel("GPD")
plt.ylabel("$St$")
plt.xlim([9,61])
#plt.ylim([0.18,0.21])
plt.grid()
#plt.title("Widerstandsbeiwert $C_{D}$ in Abhängigkeit der Domänenbreite in Durchmessern (DpY), für Re = 200", wrap=True)

# converged values
plt.axhline(y=data[1,-1], color='tab:blue', ls=":", lw=0.8, label="FWBB BGK GPD=120")
plt.axhline(y=data[7,-1], color='tab:green', ls=":", lw=0.8, label="IBB BGK GPD=120")

literature = [0.2,0.192,0.196,0.202,0.195,0.201,0.192,0.191]
# plt.axhline(y=0.2, color="r", ls="-.", lw=0.5, label="literature")
# for lit in literature[1:]:
#     plt.axhline(y=lit, color="r", ls="-.", lw=0.5)
for lit in literature:
    plt.axhline(y=lit, color="r", ls="", marker="", lw=0.5)
ylim_lock = ax1.set_ylim()
plt.axhline(y=100, color="r", ls="", marker="_",lw=0.5, label="literature")

ax2 = ax1.twinx()
ax2.set_yticks(literature, labels=[" "]*len(literature))
ax2.set_ylim(ylim_lock)
ax1.set_ylim(ylim_lock)
ax2.tick_params(color='r', direction='in', width=1.2)

ax1.legend(fontsize="6")
plt.savefig(folder+"/plots/"+name+".png")
plt.show()

