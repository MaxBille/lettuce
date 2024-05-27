import numpy as np
import tikzplotlib as tikz
import matplotlib.pyplot as plt

if 0:
    lang = ["Average", "Maximum", "Minimum", "Standard deviation"]
    kurz = ["avg", "max", "min", "std"]

    # "pcoeff_average","pcoeff_maximum","pcoeff_minimum","pcoeff_stddev","arc_length","Points:0","Points:1","Points:2"


    values_ohne = []
    values_mit = []
    angles = [20, 35, 50]

    for j in range(0,3):
            a = np.genfromtxt(f"/media/martin/extSSD/house_csv/house_overhang_{angles[j]}_1.csv", delimiter=',')
            b = np.genfromtxt(f"/media/martin/extSSD/house_csv/house_overhang_{angles[j]}_2.csv", delimiter=',')
            c = np.concatenate([a[1:, :],b[1:, :]], axis=0)
            c[:, -4] = (c[:, -3] - np.min(c[:, -3])) / (np.max(c[:, -3]) - np.min(c[:, -3]))
            offset = np.min(c[:, -3])
            div = (np.max(c[:, -3]) - np.min(c[:, -3]))
            values_mit.append(c)

    for j in range(0,3):
            a = np.genfromtxt(f"/media/martin/extSSD/house_csv/house_{angles[j]}_1.csv", delimiter=',')
            b = np.genfromtxt(f"/media/martin/extSSD/house_csv/house_{angles[j]}_2.csv", delimiter=',')
            c = np.concatenate([a[1:, :],b[1:, :]], axis=0)
            c[:, -4] = (c[:, -3] - offset) / div
            values_ohne.append(c)

    for i in range(0, 3):
        for j in range(0, 4):
            plt.figure()
            plt.plot(values_ohne[i][:, -4], values_ohne[i][:, j], label=f"no overhang")
            plt.plot(values_mit[i][:, -4], values_mit[i][:, j], label=f"overhang")
            plt.grid()
            plt.xlabel("Position along roof surface")
            plt.ylabel(f"{lang[j]} of $C_p$")
            plt.legend()
            plt.title(f"{angles[i]}° roof pitch")
            tikz.clean_figure()
            tikz.save(f"house_overhang_p_{kurz[j]}_{angles[i]}.tikz")
            plt.show()

if 0:
    lang = ["Average", "Standard deviation", "Minimum"]

    angles = [20, 35, 50]
    for h in range(0, 3): # avg, std, min
        fig, axes = plt.subplots(1, 3, constrained_layout=True)
        for i in range(0, 3): # winkel
            p = np.load(f"/home/martin/Ablage/p_auswertung/p_house_House_Final_0_{angles[i]}.npy") / (
                        0.5 * 1.225 * 0.051985464155929494 ** 2)
            p_mit = np.load(f"/home/martin/Ablage/p_auswertung/p_house_House_Final_6_{angles[i]}.npy") / (
                    0.5 * 1.225 * 0.051985464155929494 ** 2)
            offset = (len(p_mit[0, 2:]) - len(p[0, 2:]))/2
            print(offset)
            offset = int(offset)
            if h == 0:
                axes[i].plot(np.linspace(0, 1, len(p_mit[0, 2:]))[offset:len(p[0,2:])+offset], np.mean(p[100000:, 2:], axis=0), label=f"normal")
                axes[i].plot(np.linspace(0, 1, len(p_mit[0, 2:])), np.mean(p_mit[100000:, 2:], axis=0), label=f"overhang")
            elif h == 1:
                axes[i].plot(np.linspace(0, 1, len(p_mit[0, 2:]))[offset:len(p[0, 2:]) + offset],
                             np.std(p[100000:, 2:], axis=0), label=f"normal")
                axes[i].plot(np.linspace(0, 1, len(p_mit[0, 2:])), np.std(p_mit[100000:, 2:], axis=0), label=f"overhang")
            else:
                axes[i].plot(np.linspace(0, 1, len(p_mit[0, 2:]))[offset:len(p[0, 2:]) + offset],
                             np.min(p[100000:, 2:], axis=0), label=f"normal")
                axes[i].plot(np.linspace(0, 1, len(p_mit[0, 2:])), np.min(p_mit[100000:, 2:], axis=0), label=f"overhang")
        for i in range(0, 3):
            axes[i].legend()
            axes[i].grid()
            axes[i].set_xlabel("Position along roof surface")
            axes[i].set_ylabel(f"{lang[h]} of $C_p$")
            axes[i].set_title(f"{angles[i]}° roof pitch")

        tikz.clean_figure()
        tikz.save(f"house_p_overhang_{lang[h]}.tikz")
        plt.show()

if 0:
    values_ohne = []
    values_mit = []
    angles = [20, 35, 50]

    for j in range(0,3):
        a = np.genfromtxt(f"/media/martin/extSSD/house_csv/house_overhang_{angles[j]}_1.csv", delimiter=',')
        b = np.genfromtxt(f"/media/martin/extSSD/house_csv/house_overhang_{angles[j]}_2.csv", delimiter=',')
        c = np.concatenate([a[1:, :],b[1:, :]], axis=0)
        c[:, -4] = (c[:, -3] - np.min(c[:, -3])) / (np.max(c[:, -3]) - np.min(c[:, -3]))
        offset = np.min(c[:, -3])
        div = (np.max(c[:, -3]) - np.min(c[:, -3]))
        values_mit.append(c)

    lang = ["Average", "Standard deviation", "Minimum"]

    angles = [20, 35, 50]
    fig, axes = plt.subplots(1, 3)

    i = 1
    p_mit = np.load(f"/home/martin/Ablage/p_auswertung/p_house_House_Final_6_{angles[i]}.npy") / (0.5 * 1.225 * 0.051985464155929494 ** 2)
    axes[0].plot(np.linspace(0, 1, len(p_mit[0, 2:])), np.mean(p_mit[100000:, 2:], axis=0), label=f"final parameters")
    axes[1].plot(np.linspace(0, 1, len(p_mit[0, 2:])), np.std(p_mit[100000:, 2:], axis=0), label=f"final parameters")
    axes[2].plot(np.linspace(0, 1, len(p_mit[0, 2:])), np.min(p_mit[100000:, 2:], axis=0), label=f"final parameters")

    axes[0].plot(values_mit[i][:, -4], values_mit[i][:, 0], label=f"other")
    axes[1].plot(values_mit[i][:, -4], values_mit[i][:, 3], label=f"other")
    axes[2].plot(values_mit[i][:, -4], values_mit[i][:, 2], label=f"other")

    for j in range(0, 3):
        axes[j].legend()
        axes[j].grid()
        axes[j].set_xlabel("Position along roof surface")
        axes[j].set_ylabel(f"{lang[j]} of $C_p$")
        axes[j].set_title(f"{lang[j]}")

    tikz.clean_figure()
    tikz.save(f"house_p_overhang_paramComp.tikz")
    plt.show()