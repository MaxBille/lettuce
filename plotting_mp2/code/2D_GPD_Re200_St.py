import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({'font.size': 9}) # font size was 11
matplotlib.rcParams.update({'lines.linewidth': 0.8})
matplotlib.rcParams.update({'figure.figsize': [6,3]})  # 4,3 war vorher / 3x [3.2,2] / Gesamt 6.202inches textwidth
matplotlib.rcParams.update({'figure.autolayout': True})

# data source
folder = "/home/mbille/lettuce/plotting_mp2"  # HBRS
#folder =   #BONN

name = "2D_GPD_Re200_St"

data = np.genfromtxt(folder+"/data/"+name+".csv", delimiter=",")
# PARAMETERS: Re200, DpY19, D2Q9

fwbb_bgk = plt.plot(data[0], data[1], ls="--", lw=1, marker=".", color="tab:blue", label="FWBB BGK")
#fwbb_reg = plt.plot(data[0], data[2], ls="--", lw=1, marker="^", color="tab:blue", label="FWBB REG")
fwbb_kbc = plt.plot(data[0], data[3], ls="--", lw=1, marker="x", color="tab:blue", label="FWBB KBC")

hwbb_bgk = plt.plot(data[0], data[4], ls="--", lw=1, marker=".", color="tab:orange", label="HWBB BGK")
#hwbb_reg = plt.plot(data[0], data[5], ls="--", lw=1, marker="^", color="tab:orange", label="HWBB REG")
hwbb_kbc = plt.plot(data[0], data[6], ls="--", lw=1, marker="x", color="tab:orange", label="HWBB KBC")

ibb_bgk = plt.plot(data[0], data[7], ls="--", lw=1, marker=".", color="tab:green", label="IBB BGK")
#ibb_reg = plt.plot(data[0], data[8], ls="--", lw=1, marker="^", color="tab:green", label="IBB REG")
ibb_kbc = plt.plot(data[0], data[9], ls="--", lw=1, marker="x", color="tab:green", label="IBB KBC")

plt.xlabel("GPD")
plt.ylabel("$St$")
plt.xlim([9,62])
#plt.ylim([0.18,0.21])
plt.grid()
#plt.title("Widerstandsbeiwert $C_{D}$ in Abhängigkeit der Domänenbreite in Durchmessern (DpY), für Re = 200", wrap=True)

plt.axhline(y=data[1,-1], color='tab:blue', ls="-", lw=0.5, label="FWBB_BGK_GPD120")
plt.axhline(y=data[7,-1], color='tab:green', ls="-", lw=0.5, label="IBB_BGK_GPD120")

literature = [0.2,0.192,0.196,0.202,0.195,0.201,0.192,0.191]
plt.axhline(y=0.2, color="r", ls="-.", lw=0.5, label="literature")
for lit in literature[1:]:
    plt.axhline(y=lit, color="r", ls="-.", lw=0.5)
plt.legend()
plt.savefig(folder+"/plots/"+name+".png")
plt.show()

