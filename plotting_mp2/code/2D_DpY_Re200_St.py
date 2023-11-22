import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# matplotlib.rcParams.update({'font.size': 11})
# matplotlib.rcParams.update({'lines.linewidth': 0.8})
# matplotlib.rcParams.update({'figure.figsize': [4,3]})  # 3x [3.2,2] / Gesamt 6.202inches textwidth
# matplotlib.rcParams.update({'figure.autolayout': True})
# matplotlib.rcParams.update({'figure.dpi': 300})
matplotlib.style.use('../figure_style_2column_dualplot.mplstyle')
matplotlib.rcParams.update({'lines.linestyle': '--'})

# data source
folder = "/home/mbille/lettuce/plotting_mp2"  # HBRS
#folder =   #BONN

name = "2D_DpY_Re200_St"

data = np.genfromtxt(folder+"/data/"+name+".csv", delimiter=",")
# PARAMETERS: Re200, GPD30, D2Q9

fig, ax1 = plt.subplots()

fwbb_bgk = plt.plot(data[0][np.isfinite(data[1])], data[1][np.isfinite(data[1])], marker=".", color="tab:blue", label="FWBB BGK")
fwbb_kbc = plt.plot(data[0][np.isfinite(data[2])], data[2][np.isfinite(data[2])], marker="x", color="tab:blue", label="FWBB KBC")
hwbb_bgk = plt.plot(data[0][np.isfinite(data[3])], data[3][np.isfinite(data[3])], marker=".", color="tab:orange", label="HWBB BGK")
hwbb_kbc = plt.plot(data[0][np.isfinite(data[4])], data[4][np.isfinite(data[4])], marker="x", color="tab:orange", label="HWBB KBC")
ibb_bgk = plt.plot(data[0][np.isfinite(data[5])], data[5][np.isfinite(data[5])], ls='', marker=".", color="tab:green", label="IBB BGK")
ibb_kbc = plt.plot(data[0][np.isfinite(data[6])], data[6][np.isfinite(data[5])], marker="x", color="tab:green", label="IBB KBC")

fit=False
if fit:
    ## TEST FIT exponential for ibb_bgk
    import scipy
    def exp_func(xx,a,b,c):
        return a*np.exp(b*xx)+c

    colors = ["tab:blue", "tab:blue", "tab:orange", "tab:orange", "tab:green", "tab:green"]
    data_num = [
        1,
        2,
     #   3,
     #   4,
        5,
        6,
    ]
    for i in data_num:
        data_x, data_y = data[0][np.isfinite(data[i])], data[i][np.isfinite(data[i])]
        coefficients, values = scipy.optimize.curve_fit(exp_func, data_x, data_y, p0=(5, -1, 0))
        plt.plot(np.linspace(10,200,191), exp_func(np.linspace(10,200,191), *coefficients), ls="-", marker="", color=colors[i-1])

# x = data[:, ~np.isnan(data).any(axis=0)]
# #x = x[0]
# #x= data[0,:]
# y = data[:, ~np.isnan(data).any(axis=0)]
# #y = y[2]
# #y=data[5,:]
# # def exp_func(xx,a,b,c):
# #     return a*np.exp(b*xx)+c
# def hyper_func(xx,a,b,c):
#     return 1/(a*xx**2+b*xx+c)
# #coefficients, values = scipy.optimize.curve_fit(exp_func, x, y, p0=(5,-1,0))
# #coefficients, values = scipy.optimize.curve_fit(hyper_func, x, y)
# print(coefficients)
#plt.plot(x, exp_func(x, *coefficients), 'r-')
#plt.plot(x, hyper_func(x, *coefficients), 'r-')

plt.xlabel("D/Y")
plt.ylabel("$St$")
plt.xlim([0,205])
#plt.ylim([0.18,0.21])
plt.grid()
#plt.title("Widerstandsbeiwert $C_{D}$ in Abhängigkeit der Domänenbreite in Durchmessern (DpY), für Re = 200", wrap=True)

literature = [0.2,0.192,0.196,0.202,0.195,0.201,0.192,0.191]
for lit in literature:
    plt.axhline(y=lit, color="r", ls="", marker="", lw=0.5)
ax2 = ax1.twinx()
ax2.set_yticks(literature, labels=[" "]*len(literature))
ax2.set_ylim(ax1.set_ylim())
ax2.tick_params(color='r', direction='in', width=1.2)
#plt.legend()
plt.savefig(folder+"/plots/"+name+".png")
plt.show()

