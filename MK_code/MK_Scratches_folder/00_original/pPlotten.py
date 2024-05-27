import numpy as np
import tikzplotlib as tikz
import matplotlib.pyplot as plt

angleList = []
for angle in [20, 35, 50]:
    mioListe = []
    for over in ["mit", "ohne"]:
        input = []
        for i in range(1,3):
            input.append(np.genfromtxt(f"/mnt/hgfs/VMshare/The last plot/p_{over}_{angle}_{i}.csv", delimiter=','))
        input[1][:, 4] = input[1][:, 4] + input[0][-1, 4]
        mioListe.append(np.vstack([input[0], input[1]]))
        if over == "ohne":
            #mioListe[-1][:, 4] = mioListe[-1][:, 4] / (60 / np.cos(angle / 180 * np.pi))
            if angle == 50:
                mioListe[-1][:, 4] = mioListe[-1][:, 4]  + (6 / np.cos(angle / 180 * np.pi))
            else:
                mioListe[-1][:, 4] = mioListe[-1][:, 4] + (5 / np.cos(angle / 180 * np.pi))
        if angle == 50:
            mioListe[-1][:, 4] = mioListe[-1][:, 4] / (72 /np.cos(angle/180*np.pi))
        else:
            mioListe[-1][:, 4] = mioListe[-1][:, 4] / (70 / np.cos(angle / 180 * np.pi))
        mioListe[-1][:, 0:4] = mioListe[-1][:, 0:4] / (0.5 * 1.225 * 0.029705979517673998**2)
    angleList.append(mioListe)
# "p_average","p_maximum","p_minimum","p_stddev","arc_length","Points:0","Points:1","Points:2"

if 0:
    for i in range(0, 3):
        plt.plot(angleList[i][0][:, 4], angleList[i][0][:, 0], label=f"{[20, 35, 50][i]}°")

    plt.grid()
    plt.xlabel("Position auf der Dachoberfläche")
    plt.ylabel("$C_p$")
    plt.legend()
    plt.title(f"Mittlerer Druckkoeffizient")
    tikz.clean_figure()
    tikz.save(f"/mnt/hgfs/VMshare/The last plot/avg_ohne.tikz")
    plt.show()

    for i in range(0, 3):
        plt.plot(angleList[i][0][:, 4], angleList[i][0][:, 1], label=f"{[20, 35, 50][i]}°")

    plt.grid()
    plt.xlabel("Position auf der Dachoberfläche")
    plt.ylabel("$C_p$")
    plt.legend()
    plt.title(f"Maxima des Druckkoeffizienten")
    tikz.clean_figure()
    tikz.save(f"/mnt/hgfs/VMshare/The last plot/max_ohne.tikz")
    plt.show()

    for i in range(0, 3):
        plt.plot(angleList[i][0][:, 4], angleList[i][0][:, 2], label=f"{[20, 35, 50][i]}°")

    plt.grid()
    plt.xlabel("Position auf der Dachoberfläche")
    plt.ylabel("$C_p$")
    plt.legend()
    plt.title(f"Minima des Druckkoeffizienten")
    tikz.clean_figure()
    tikz.save(f"/mnt/hgfs/VMshare/The last plot/min_ohne.tikz")
    plt.show()

    for i in range(0, 3):
        plt.plot(angleList[i][0][:, 4], angleList[i][0][:, 3], label=f"{[20, 35, 50][i]}°")

    plt.grid()
    plt.xlabel("Position auf der Dachoberfläche")
    plt.ylabel("$C_p$")
    plt.legend()
    plt.title(f"Standardabweichung des Druckkoeffizienten")
    tikz.clean_figure()
    tikz.save(f"/mnt/hgfs/VMshare/The last plot/std_ohne.tikz")
    plt.show()

lang = ["Mittelwert", "Maximum", "Minimum", "Standardabweichung"]
kurz = ["avg", "max", "min", "std"]

for j in range(0, 4):
    for i, angle in enumerate([20, 35, 50]):
        plt.plot(angleList[i][0][:, 4], angleList[i][0][:,j], label = "mit Überstand")
        plt.plot(angleList[i][1][:, 4], angleList[i][1][:, j], label = "ohne Überstand")
        plt.grid()
        plt.xlabel("Position auf der Dachoberfläche")
        plt.ylabel(f"{lang[j]} von $C_p$")
        plt.legend()
        plt.title(f"{lang[j]}")
        tikz.clean_figure()
        tikz.save(f"/mnt/hgfs/VMshare/The last plot/mitohne_{angle}_{kurz[j]}.tikz")
        plt.show()