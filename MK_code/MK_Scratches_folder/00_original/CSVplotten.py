import numpy as np
from matplotlib import pyplot as plt
import tikzplotlib as tikz


diffz = []
txt = [167, 266, 369]
txt2 = [3, 5, 7]
colour = ["blue", "green", "red", "orange"]
angle = [a / 180 * np.pi for a in [16.7, 26.6, 36.9]]
offset = [-1, -0.55, 0, 0.55, 1]
for h in range(0, 3):
    u = []
    ref = []
    for i in range(1, 6):
        u.append(np.genfromtxt(f'/mnt/hgfs/VMshare/CSVs/u_{txt[h]}_{i}.csv', delimiter=','))
        ref.append(np.genfromtxt(f'/mnt/hgfs/VMshare/CSVs/u_{txt2[h]}10_{i}.csv', delimiter=','))
    for i in range(0, 5):
        u[i][:, 0] = u[i][:, 0] / 2 * 0.5 + offset[i]
        u[i][:, 1] = u[i][:, 1] / 54.5454545454545454545
        plt.plot(ref[i][:, 0], ref[i][:, 1], marker='o', color="gray", linestyle="None")
        plt.plot(u[i][1:, 0], u[i][1:, 1], color="black")
    plt.grid()
    plt.ylabel("$z/z_{Haus}$")
    plt.xlabel("$x/z_{Haus}$")
    plt.xlim(-1.5, 1.5)
    plt.ylim(0, 2.5)
    plt.title(f"Dachneigung: {[16.7, 26.6, 36.9][h]}°")
    tikz.clean_figure()
    tikz.save(f"/mnt/hgfs/VMshare/CSVs/u_{txt[h]}.tikz")
    plt.show()