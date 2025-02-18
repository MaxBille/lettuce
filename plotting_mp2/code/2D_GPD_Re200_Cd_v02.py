import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# matplotlib.rcParams.update({'font.size': 9}) # font size was 11
# matplotlib.rcParams.update({'lines.linewidth': 0.8})
# matplotlib.rcParams.update({'figure.figsize': [6,3]})  # 4,3 war vorher / 3x [3.2,2] / Gesamt 6.202inches textwidth
# matplotlib.rcParams.update({'figure.autolayout': True})

# matplotlib.rcParams.update({'font.size': 7}) # font size was 11
# matplotlib.rcParams.update({'lines.linewidth': 0.8})
# #matplotlib.rcParams.update({'figure.figsize': [6,3]})  # 4,3 war vorher / 3x [3.2,2] / Gesamt 6.202inches textwidth
# matplotlib.rcParams.update({'figure.figsize': [3.4876,3.4876*0.65]})
# matplotlib.rcParams.update({'figure.autolayout': True})
# matplotlib.rcParams.update({'figure.dpi': 300})

matplotlib.style.use('../figure_style_2column_singleplot.mplstyle')
matplotlib.rcParams.update({'lines.linestyle': '--'})

# for tick-str formatting
from matplotlib.ticker import FormatStrFormatter

# data source
folder = "/home/mbille/lettuce/plotting_mp2"  # HBRS
#folder =   #BONN

name = "2D_GPD_Re200_Cd_v02"

data = np.genfromtxt(folder+"/data/"+name+".csv", delimiter=",")
# PARAMETERS: Re200, DpY19, D2Q9

gpd_2_index = np.where(np.isin(data[0], [2,4,8,16,32,64]))#,128]))
gpd_3_index = np.where(np.isin(data[0], [3,6,12,24,48,96]))
gpd_5_index = np.where(np.isin(data[0], [5,10,20,40,80]))

converged_mean = np.nanmean(data[1:,-1])
converged_mean = data[9,-1]
converged_mean = data[4,-1]
print(converged_mean)
data_delta = np.abs(data-converged_mean)
#print(data_delta)

#TODO: y-ticks "schön" setzen
#TODO: x-ticks "schön" setzen
#TODO; sinnvolles Grid setzen -> mal sehen, wie andere Leute ihre LogLog Plots in x und y skalieren!
#TODO: Text: wenn man hier konventionell mal nur Verdoppelt ab Basis 2, 3, oder 5 ("multiples of base ..."), dann sieht man manchmal eine Konvergenz, manchmal nicht
# im Endeffekt wird genau mein rechts Bild aus der DSFD-Präsentation erzeugt, nur halt mit 2^x, 3^x und 5^x als Auflösungswerte der x-Achse

#TODO: das gleiche wie für fwbb_bgk auch noch für die beiden anderen BC und CO
# Darstellung über: Farbe, Symbol, linestyle? oder anstelle Linestyle einfach einzelne Plots pro BC, dann hab ich drei Plots.
# OPT: Gerade 1/2^x oderso einbringen? damit man sieht,d ass das parallel ist?

### COMPELTE Plot
fig_all, ax_all = plt.subplots()

fwbb_bgk = plt.semilogx(data[0], data[1], marker=".", color="tab:blue", label="FWBB BGK")
#fwbb_reg = plt.loglog(data[0], data[2], marker="^", color="tab:blue", label="FWBB REG")
fwbb_kbc = plt.semilogx(data[0], data[3], marker="x", color="tab:blue", label="FWBB KBC")

hwbb_bgk = plt.semilogx(data[0], data[4], marker=".", color="tab:orange", label="HWBB BGK")
#hwbb_reg = plt.loglog(data[0], data[5], marker="^", color="tab:orange", label="HWBB REG")
hwbb_kbc = plt.semilogx(data[0], data[6], marker="x", color="tab:orange", label="HWBB KBC")

ibb_bgk = plt.semilogx(data[0], data[7], marker=".", color="tab:green", label="IBB BGK")
#ibb_reg = plt.loglog(data[0], data[8], marker="^", color="tab:green", label="IBB REG")
ibb_kbc = plt.semilogx(data[0], data[9], marker="x", color="tab:green", label="IBB KBC")

plt.xlabel("GPD")
plt.ylabel("$C_{D}$")
#plt.xlim([9,61])
#plt.ylim([0.18,0.21])
plt.grid()
#plt.title("Widerstandsbeiwert $C_{D}$ in Abhängigkeit der Domänenbreite in Durchmessern (DpY), für Re = 200", wrap=True)

# converged values as dotted lines
plt.axhline(y=data[1,-1], color='tab:blue', ls=":", lw=0.8, label="FWBB BGK GPD=128")
plt.axhline(y=data[7,-1], color='tab:green', ls=":", lw=0.8, label="IBB BGK GPD=128")

# add  red literature ticks on RIGHT side
literature = [1.4,1.31,1.19,1.31,1.33,1.172,1.29,1.45,1.26,1.36,1.4087]
# plt.axhline(y=1.4, color="r", ls="-.", lw=0.5, label="literature")
# for lit in literature[1:]:
#     plt.axhline(y=lit, color="r", ls="-.", lw=0.5)
for lit in literature:
    plt.axhline(y=lit, color="r", ls="", marker="", lw=0.5)
ax_all2 = ax_all.twinx()
ax_all2.set_yticks(literature, labels=[" "]*len(literature))
ax_all2.set_ylim(ax_all.set_ylim())
ax_all2.tick_params(color='r', direction='in', width=1.2)

# format y-axis labels
ax_all.yaxis.set_major_formatter(FormatStrFormatter('%.1f  '))
# plt.savefig(folder+"/plots/"+name+"_complete_noLegend.png")


plt.savefig(folder+"/plots/"+name+"_complete_noLegend.png")

# ax_all.legend(fontsize="5")
# plt.savefig(folder+"/plots/"+name+"_complete.png")

plt.show()

#############################################
### loglog-Plots

##FWBB
fig_loglog_fwbb, ax_fwbb = plt.subplots()
# BC (plot), CO (marker), base (color

fwbb_bgk_base2 = plt.loglog(data[0][gpd_2_index], data_delta[1][gpd_2_index], marker=".", color="tab:blue", label="FWBB BGK base2")
fwbb_bgk_base3 = plt.loglog(data[0][gpd_3_index], data_delta[1][gpd_3_index], marker=".", color="tab:orange", label="FWBB BGK base3")
fwbb_bgk_base5 = plt.loglog(data[0][gpd_5_index], data_delta[1][gpd_5_index], marker=".", color="tab:green", label="FWBB BGK base5")

fwbb_kbc_base2 = plt.loglog(data[0][gpd_2_index], data_delta[3][gpd_2_index], marker="x", color="tab:blue", label="FWBB KBC base2")
fwbb_kbc_base3 = plt.loglog(data[0][gpd_3_index], data_delta[3][gpd_3_index], marker="x", color="tab:orange", label="FWBB KBC base3")
fwbb_kbc_base5 = plt.loglog(data[0][gpd_5_index], data_delta[3][gpd_5_index], marker="x", color="tab:green", label="FWBB KBC base5")

plt.xlabel("GPD")
plt.ylabel(r"$|C_{D} - \overline{C_{D,GPD128}}|$")
#plt.xlim([9,61])
#plt.ylim([0.18,0.21])
plt.grid()
#plt.title("Widerstandsbeiwert $C_{D}$ in Abhängigkeit der Domänenbreite in Durchmessern (DpY), für Re = 200", wrap=True)

plt.legend()
plt.savefig(folder+"/plots/"+name+"_loglog_fwbb.png")
plt.show()

##HWBB
fig_loglog_hwbb, ax_hwbb = plt.subplots()
# BC (plot), CO (marker), base (color

hwbb_bgk_base2 = plt.loglog(data[0][gpd_2_index], data_delta[4][gpd_2_index], marker=".", color="tab:blue", label="HWBB BGK base2")
hwbb_bgk_base3 = plt.loglog(data[0][gpd_3_index], data_delta[4][gpd_3_index], marker=".", color="tab:orange", label="HWBB BGK base3")
hwbb_bgk_base5 = plt.loglog(data[0][gpd_5_index], data_delta[4][gpd_5_index], marker=".", color="tab:green", label="HWBB BGK base5")

hwbb_kbc_base2 = plt.loglog(data[0][gpd_2_index], data_delta[6][gpd_2_index], marker="x", color="tab:blue", label="HWBB KBC base2")
hwbb_kbc_base3 = plt.loglog(data[0][gpd_3_index], data_delta[6][gpd_3_index], marker="x", color="tab:orange", label="HWBB KBC base3")
hwbb_kbc_base5 = plt.loglog(data[0][gpd_5_index], data_delta[6][gpd_5_index], marker="x", color="tab:green", label="HWBB KBC base5")

plt.xlabel("GPD")
plt.ylabel(r"$|C_{D} - \overline{C_{D,GPD128}}|$")
#plt.xlim([9,61])
#plt.ylim([0.18,0.21])
plt.grid()
#plt.title("Widerstandsbeiwert $C_{D}$ in Abhängigkeit der Domänenbreite in Durchmessern (DpY), für Re = 200", wrap=True)

plt.legend()
plt.savefig(folder+"/plots/"+name+"_loglog_hwbb.png")
plt.show()

##IBB
fig_loglog_ibb, ax_ibb = plt.subplots()
# BC (plot), CO (marker), base (color

ibb_bgk_base2 = plt.loglog(data[0][gpd_2_index], data_delta[7][gpd_2_index], marker=".", color="tab:blue", label="IBB BGK base2")
ibb_bgk_base3 = plt.loglog(data[0][gpd_3_index], data_delta[7][gpd_3_index], marker=".", color="tab:orange", label="IBB BGK base3")
ibb_bgk_base5 = plt.loglog(data[0][gpd_5_index], data_delta[7][gpd_5_index], marker=".", color="tab:green", label="IBB BGK base5")

ibb_kbc_base2 = plt.loglog(data[0][gpd_2_index], data_delta[9][gpd_2_index], marker="x", color="tab:blue", label="IBB KBC base2")
ibb_kbc_base3 = plt.loglog(data[0][gpd_3_index], data_delta[9][gpd_3_index], marker="x", color="tab:orange", label="IBB KBC base3")
ibb_kbc_base5 = plt.loglog(data[0][gpd_5_index], data_delta[9][gpd_5_index], marker="x", color="tab:green", label="IBB KBC base5")

plt.xlabel("GPD")
plt.ylabel(r"$|C_{D} - \overline{C_{D,GPD128}}|$")
#plt.xlim([9,61])
#plt.ylim([0.18,0.21])
plt.grid()
#plt.title("Widerstandsbeiwert $C_{D}$ in Abhängigkeit der Domänenbreite in Durchmessern (DpY), für Re = 200", wrap=True)

plt.legend()
plt.savefig(folder+"/plots/"+name+"_loglog_ibb.png")
plt.show()

##EXTRA: IBB+KBC only
fig_loglog_ibbkbc, ax_ibbkbc = plt.subplots()
# BC (plot), CO (marker), base (color

# ibb_bgk_base2 = plt.loglog(data[0][gpd_2_index], data_delta[7][gpd_2_index], marker=".", color="tab:blue", label="IBB BGK base2")
# ibb_bgk_base3 = plt.loglog(data[0][gpd_3_index], data_delta[7][gpd_3_index], marker=".", color="tab:orange", label="IBB BGK base3")
# ibb_bgk_base5 = plt.loglog(data[0][gpd_5_index], data_delta[7][gpd_5_index], marker=".", color="tab:green", label="IBB BGK base5")

ibb_kbc_base2 = plt.loglog(data[0][gpd_2_index], data_delta[9][gpd_2_index], marker="x", color="tab:blue", label="base res. 16 ( = 2 x 2⁴)")
ibb_kbc_base3 = plt.loglog(data[0][gpd_3_index], data_delta[9][gpd_3_index], marker="x", color="tab:orange", label="base res. 12 ( = 3 x 2²)")
ibb_kbc_base5 = plt.loglog(data[0][gpd_5_index], data_delta[9][gpd_5_index], marker="x", color="tab:green", label="base res. 10 ( = 5 x 2¹)")

plt.xlabel("GPD")
plt.ylabel(r"$|C_{D} - \overline{C_{D,GPD128}}|$")
#plt.xlim([9,61])
#plt.ylim([10**-3,10**-1])
plt.grid()
#plt.title("Widerstandsbeiwert $C_{D}$ in Abhängigkeit der Domänenbreite in Durchmessern (DpY), für Re = 200", wrap=True)

plt.legend()
plt.savefig(folder+"/plots/"+name+"_loglog_ibbkbc.png")
plt.show()


##HWBBBGK
fig_loglog_hwbbbgk, ax_hwbbbgk = plt.subplots()
# BC (plot), CO (marker), base (color

hwbb_bgk_base2 = plt.loglog(data[0][gpd_2_index], data_delta[4][gpd_2_index], marker=".", color="tab:blue", label="HWBB BGK base2")
hwbb_bgk_base3 = plt.loglog(data[0][gpd_3_index], data_delta[4][gpd_3_index], marker=".", color="tab:orange", label="HWBB BGK base3")
hwbb_bgk_base5 = plt.loglog(data[0][gpd_5_index], data_delta[4][gpd_5_index], marker=".", color="tab:green", label="HWBB BGK base5")

# hwbb_kbc_base2 = plt.loglog(data[0][gpd_2_index], data_delta[6][gpd_2_index], marker="x", color="tab:blue", label="HWBB KBC base2")
# hwbb_kbc_base3 = plt.loglog(data[0][gpd_3_index], data_delta[6][gpd_3_index], marker="x", color="tab:orange", label="HWBB KBC base3")
# hwbb_kbc_base5 = plt.loglog(data[0][gpd_5_index], data_delta[6][gpd_5_index], marker="x", color="tab:green", label="HWBB KBC base5")

plt.xlabel("GPD")
plt.ylabel(r"$|C_{D} - \overline{C_{D,GPD128}}|$")
#plt.xlim([9,61])
plt.ylim([10**-3,10**-0])
plt.grid()
#plt.title("Widerstandsbeiwert $C_{D}$ in Abhängigkeit der Domänenbreite in Durchmessern (DpY), für Re = 200", wrap=True)

plt.legend()
plt.savefig(folder+"/plots/"+name+"_loglog_hwbbbgk_no128.png")
plt.show()

##HWBBBGK + ALL
fig_loglog_hwbbbgk, ax_hwbbbgk = plt.subplots()
# BC (plot), CO (marker), base (color

hwbb_bgk = plt.loglog(data[0], data_delta[4], linestyle="", marker="x", color="grey", label="all tested res.", alpha=0.8)

hwbb_bgk_base2 = plt.loglog(data[0][gpd_2_index], data_delta[4][gpd_2_index], marker=".", color="tab:blue", label="base res. 16 ( = 2 x 2⁴)")
hwbb_bgk_base3 = plt.loglog(data[0][gpd_3_index], data_delta[4][gpd_3_index], marker=".", color="tab:orange", label="base res. 12 ( = 3 x 2²)")
hwbb_bgk_base5 = plt.loglog(data[0][gpd_5_index], data_delta[4][gpd_5_index], marker=".", color="tab:green", label="base res. 10 ( = 5 x 2¹)")



# hwbb_kbc_base2 = plt.loglog(data[0][gpd_2_index], data_delta[6][gpd_2_index], marker="x", color="tab:blue", label="HWBB KBC base2")
# hwbb_kbc_base3 = plt.loglog(data[0][gpd_3_index], data_delta[6][gpd_3_index], marker="x", color="tab:orange", label="HWBB KBC base3")
# hwbb_kbc_base5 = plt.loglog(data[0][gpd_5_index], data_delta[6][gpd_5_index], marker="x", color="tab:green", label="HWBB KBC base5")

plt.xlabel("GPD")
plt.ylabel(r"$|C_{D} - C_{D,GPD128}|$")
#plt.xlim([9,61])
plt.ylim([10**-3,10**-0])
plt.grid()
#plt.title("Widerstandsbeiwert $C_{D}$ in Abhängigkeit der Domänenbreite in Durchmessern (DpY), für Re = 200", wrap=True)

plt.legend()
plt.savefig(folder+"/plots/"+name+"_loglog_hwbbbgk_no128.png")
plt.show()