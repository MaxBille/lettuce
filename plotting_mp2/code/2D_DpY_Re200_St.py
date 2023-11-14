import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({'font.size': 11})
matplotlib.rcParams.update({'lines.linewidth': 0.8})
matplotlib.rcParams.update({'figure.figsize': [4,3]})  # 3x [3.2,2] / Gesamt 6.202inches textwidth
matplotlib.rcParams.update({'figure.autolayout': True})

# data source
folder = "/home/mbille/lettuce/plotting_mp2"  # HBRS
#folder =   #BONN

name = "2D_DpY_Re200_St"

data = np.genfromtxt(folder+"/data/"+name+".csv", delimiter=",")
# PARAMETERS: Re200, GPD30, D2Q9

fwbb_bgk = plt.plot(data[0], data[1], ls="--", lw=1, marker=".", color="tab:blue", label="FWBB BGK")
fwbb_kbc = plt.plot(data[0], data[2], ls="--", lw=1, marker="x", color="tab:blue", label="FWBB KBC")
hwbb_bgk = plt.plot(data[0], data[3], ls="--", lw=1, marker=".", color="tab:orange", label="HWBB BGK")
hwbb_kbc = plt.plot(data[0], data[4], ls="--", lw=1, marker="x", color="tab:orange", label="HWBB KBC")
ibb_bgk = plt.plot(data[0], data[5], ls="--", lw=1, marker=".", color="tab:green", label="IBB BGK")
ibb_kbc = plt.plot(data[0], data[6], ls="--", lw=1, marker="x", color="tab:green", label="IBB KBC")

## TEST FIT exponential for ibb_bgk
import scipy
x = data[:, ~np.isnan(data).any(axis=0)]
#x = x[0]
x= data[0,:6]
y = data[:, ~np.isnan(data).any(axis=0)]
#y = y[2]
y=data[5,:6]
def exp_func(xx,a,b,c):
    return a*np.exp(b*xx)+c
def hyper_func(xx,a,b,c):
    return 1/(a*xx**2+b*xx+c)
coefficients, values = scipy.optimize.curve_fit(exp_func, x, y, p0=(5,-1,0))
#coefficients, values = scipy.optimize.curve_fit(hyper_func, x, y)
print(coefficients)
plt.plot(x, exp_func(x, *coefficients), 'r-')
#plt.plot(x, hyper_func(x, *coefficients), 'r-')

plt.xlabel("D/Y")
plt.ylabel("$St$")
plt.xlim([0,205])
plt.ylim([0.18,0.21])
plt.grid()
#plt.title("Widerstandsbeiwert $C_{D}$ in Abhängigkeit der Domänenbreite in Durchmessern (DpY), für Re = 200", wrap=True)

literature = [0.2,0.192,0.196,0.202,0.195,0.201,0.192,0.191]
plt.axhline(y=0.2, color="r", ls="-.", lw=0.5, label="literature")
for lit in literature[1:]:
    plt.axhline(y=lit, color="r", ls="-.", lw=0.5)
#plt.legend()
plt.savefig(folder+"/plots/"+name+".png")
plt.show()

