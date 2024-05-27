import numpy as np
from matplotlib import  pyplot as plt
import tikzplotlib as tikz

if 0:
    offset = [-1, -0.55, 0, 0.55, 1]
    diffz = []
    txt = [167, 266, 369]
    txt2 = [3, 5, 7]
    for h in range(0, 3):
        u = []
        ref = []
        for i in range(1, 6):
            u.append(np.genfromtxt(f'/home/martin/Ablage/p_auswertung/k_{txt[h]}_{i}.csv', delimiter=','))
            ref.append(np.genfromtxt(f'/home/martin/Ablage/p_auswertung/k_{txt2[h]}10_{i}.csv', delimiter=','))
        for i in range(0, 5):
            ref[i][:, 0] = (-ref[i][:, 0] - offset[i]) * 0.4 # 4 für u
            u[i][:, 1] = u[i][:, 1] / 54.5454545454545454545
            plt.plot(ref[i][:, 0], ref[i][:, 1], "bo")
            plt.plot(u[i][:, 0], u[i][:, 1])
            #plt.show()

        diff = []
        for i in range(4, 5):
            for j in range(len(ref[i])):
                pos = np.where((np.abs(u[i][:, 1]-ref[i][j, 1]) == np.min(np.abs(u[i][:, 1]-ref[i][j, 1]))))
                if (np.abs(u[i][pos, 0] - ref[i][j, 0])/ref[i][j, 0]) < 3:
                    diff.append(np.abs(u[i][pos, 0] - ref[i][j, 0])/ref[i][j, 0])
                else:
                    print(f"i {i}, j {j}, h {h}")

        print(f"Mittlere Abweichung: {np.mean(diff)}, maximale Abweichung: {np.max(diff)}")
        diffz.append(np.asarray(diff).squeeze())


    print(f"Globale Abweichung: {np.mean(np.hstack(diffz))}")
    plt.boxplot(diffz)
    plt.show()


if 1:
    diffz = []
    txt = [167, 266, 369]
    txt2 = [3, 5, 7]
    colour = ["blue", "green", "red", "orange"]
    angle = [a / 180 * np.pi for a in [16.7, 26.6, 36.9]]
    for h in range(0, 3):
        p = []
        ref = []
        for i in range(1, 5):
            p.append(np.genfromtxt(f'/mnt/hgfs/VMshare/CVSs ohne Doofheit/p_{txt[h]}_{i}.csv', delimiter=','))
            ref.append(np.genfromtxt(f'/home/martin/Ablage/p_auswertung/p_{txt2[h]}10_{i}.csv', delimiter=','))

        diff = []
        for i in range(0, 4):
            if i == 1:
                p[i][:, 1] = p[i][:, 1] - 2/np.cos(angle[h])
            if i == 2:
                p[i][:, 1] = p[i][:, 1] - 2 * np.tan(angle[h])
            p[i][:, 1:] = p[i][:, 1:] / 54.5454545454545454545
            plt.plot(ref[i][:, 0], ref[i][:, 1], marker='o', color=colour[i], linestyle="None")
            plt.plot(p[i][:, 0], p[i][:, 1], color=colour[i])
            for j in range(len(ref[i])):
                pos = np.where((np.abs(p[i][:, 1] - ref[i][j, 1]) == np.min(np.abs(p[i][:, 1] - ref[i][j, 1]))))
                if (np.abs((p[i][pos, 0] - ref[i][j, 0]) / ref[i][j, 0])) < 3:
                    diff.append(np.abs((p[i][pos, 0] - ref[i][j, 0]) / ref[i][j, 0]))
                else:
                    print(f"i {i}, j {j}, h {h}")

        print(f"Mittlere Abweichung: {np.mean(diff)}, maximale Abweichung: {np.max(diff)}")
        diffz.append(np.asarray(diff).squeeze())
        plt.grid()
        plt.ylim(-0.098, 1.06)
        plt.ylabel("Pos. entlang der Hausoberfläche in $z_{Haus}$")
        plt.xlabel("$C_p$")
        plt.xlim(-1.19, 1.36)
        plt.title(f"Dachneigung: {[16.7, 26.6, 36.9][h]}°")
        #plt.show()
        tikz.clean_figure()
        #tikz.save(f"/mnt/hgfs/VMshare/CVSs ohne Doofheit/p_{txt[h]}.tikz")
        plt.show()
    print(f"Globale Abweichung: {np.mean(np.hstack(diffz))}")
    plt.boxplot(diffz)
    plt.show()