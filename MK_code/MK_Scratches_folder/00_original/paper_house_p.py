import numpy as np
import tikzplotlib as tikz
import matplotlib.pyplot as plt


lang = ["Average", "Maximum", "Minimum", "Standard deviation"]
kurz = ["avg", "max", "min", "std"]

# "pcoeff_average","pcoeff_maximum","pcoeff_minimum","pcoeff_stddev","arc_length","Points:0","Points:1","Points:2"


if 1:
    values = []
    angles = [20, 35, 50]
    for j in range(0,3):
            a = np.genfromtxt(f"/media/martin/extSSD/house_csv/house_overhang_{angles[j]}_1.csv", delimiter=',')
            b = np.genfromtxt(f"/media/martin/extSSD/house_csv/house_overhang_{angles[j]}_2.csv", delimiter=',')
            c = np.concatenate([a[1:, :],b[1:, :]], axis=0)
            c[:, -4] = (c[:, -3] - np.min(c[:, -3])) / (np.max(c[:, -3]) - np.min(c[:, -3]))
            values.append(c)

if 0:
    for j in range(0,4):
        plt.figure()
        for i in range(0, 3):
            plt.plot(values[i][:, -4], values[i][:, j], label=f"{angles[i]}°")
        plt.grid()
        plt.xlabel("Position along roof surface")
        plt.ylabel(f"{lang[j]} of $C_p$")
        plt.legend()
        plt.title(f"{lang[j]}")
        tikz.clean_figure()
        tikz.save(f"house_u_{kurz[j]}.tikz")
        plt.show()
if 0:

    lang = ["Average", "Standard deviation", "Minimum"]

    angles = [20, 35, 50]
    fig, axes = plt.subplots(1,3, constrained_layout=True)
    for i in range(0,3):
        p = np.load(f"/home/martin/Ablage/p_auswertung/p_house_House_Final_0_{angles[i]}.npy")/(0.5 * 1.225 * 0.051985464155929494**2)
        axes[0].plot(np.linspace(0, 1, len(p[0, 2:])), np.mean(p[100000:, 2:], axis=0), label=f"{angles[i]}°")
        axes[1].plot(np.linspace(0, 1, len(p[0, 2:])), np.std(p[100000:, 2:], axis=0), label=f"{angles[i]}°")
        #axes[1, 0].plot(np.max(p[100000:, 2:], axis=0), label=f"{angles[i]}°")
        axes[2].plot(np.linspace(0, 1, len(p[0, 2:])), np.min(p[100000:, 2:], axis=0), label=f"{angles[i]}°")


    for i in range(0,3):
        axes[i].legend()
        axes[i].grid()
        axes[i].set_xlabel("Position along roof surface")
        axes[i].set_ylabel(f"{lang[i]} of $C_p$")
        axes[i].set_title(f"{lang[i]}")

    tikz.clean_figure()
    tikz.save(f"house_p.tikz")
    plt.show()

    plt.figure()
    for i in range(0, 3):
        p = np.load(f"/home/martin/Ablage/p_auswertung/p_house_House_Final_0_{angles[i]}.npy") / (
                    0.5 * 1.225 * 0.051985464155929494 ** 2)
        plt.plot(np.linspace(0, 1, len(p[0, 2:])), np.std(p[100000:, 2:], axis=0), label=f"{angles[i]}°")

    plt.legend()
    plt.grid()
    plt.xlabel("Position along roof surface")
    plt.ylabel(f"{lang[1]} of $C_p$")

    tikz.clean_figure()
    tikz.save(f"house_p_std.tikz")
    plt.show()

if 1:
    lang = ["Average", "Standard deviation", "Minimum"]

    angles = [20, 35, 50]
    fig, axes = plt.subplots(1,3, constrained_layout=True)
    i = 1
    p = np.load(f"/home/martin/Ablage/p_auswertung/p_house_House_Final_0_{angles[i]}.npy")/(0.5 * 1.225 * 0.051985464155929494**2)
    axes[0].plot(np.linspace(0, 1, len(p[0, 2:])), np.mean(p[100000:, 2:], axis=0), label=f"final parameters")
    axes[1].plot(np.linspace(0, 1, len(p[0, 2:])), np.std(p[100000:, 2:], axis=0), label=f"final parameters")
    axes[2].plot(np.linspace(0, 1, len(p[0, 2:])), np.min(p[100000:, 2:], axis=0), label=f"final parameters")


    axes[0].plot(values[i][:, -4], values[i][:, 0], label=f"other")
    axes[1].plot(values[i][:, -4], values[i][:, 3], label=f"other")
    axes[2].plot(values[i][:, -4], values[i][:, 2], label=f"other")


    for i in range(0,3):
        axes[i].legend()
        axes[i].grid()
        axes[i].set_xlabel("Position along roof surface")
        axes[i].set_ylabel(f"{lang[i]} of $C_p$")
        axes[i].set_title(f"{lang[i]}")

    tikz.clean_figure()
    tikz.save(f"house_p_paramComp.tikz")
    plt.show()

    plt.figure()
    p = np.load(f"/home/martin/Ablage/p_auswertung/p_house_House_Final_0_{angles[1]}.npy") / (
                0.5 * 1.225 * 0.051985464155929494 ** 2)
    p2 = np.load(f"/home/martin/Ablage/p_auswertung/p_house_House_Final_0_{angles[1]}_alternativ.npy") / (
            0.5 * 1.225 * 0.051985464155929494 ** 2)
    plt.plot(np.linspace(0, 1, len(p[0, 2:])), np.mean(p[100000:, 2:], axis=0), label=f"final parameters")
    plt.plot(np.linspace(0, 1, len(p[0, 2:])), np.mean(p2[100000:, 2:], axis=0), label=f"other")
    #plt.plot(values[1][:, -4], values[1][:, 0], label=f"other")

    plt.legend()
    plt.grid()
    plt.xlabel("Position along roof surface")
    plt.ylabel(f"{lang[0]} of $C_p$")

    tikz.clean_figure()
    tikz.save(f"house_p_paramComp_avg.tikz")
    plt.show()

if 0:
    # für grid-Größen test:

    lang = ["Average", "Standard deviation", "Minimum"]

    angles = [20, 35, 50]
    fig, axes = plt.subplots(1,3)
    i = 1

    u_0 = 0.051985464155929494
    p17 = np.load("/home/martin/Ablage/anderePs/p_house_large_House_Final_0_35_part1.npy") /(0.5 * 1.225 * u_0**2)
    p17b = np.load("/home/martin/Ablage/anderePs/p_house_large_House_Final_0_35_part2.npy")/(0.5 * 1.225 * u_0**2)
    p17c = np.load("/home/martin/Ablage/anderePs/p_house_large_House_Final_0_35_part3.npy")/(0.5 * 1.225 * u_0**2)
    p17d = np.load("/home/martin/Ablage/anderePs/p_house_large_House_Final_0_35_part4.npy")/(0.5 * 1.225 * u_0**2)
    p17e = np.load("/home/martin/Ablage/anderePs/p_house_large_House_Final_0_35_part5.npy") / (0.5 * 1.225 * u_0 ** 2)
    p17f = np.load("/home/martin/Ablage/anderePs/p_house_large_House_Final_0_35_part6.npy") / (0.5 * 1.225 * u_0 ** 2)
    p_large = np.concatenate([p17, p17b, p17c, p17d, p17e, p17f], axis=0)
    print(p_large.shape)

    p = np.load(f"/home/martin/Ablage/p_auswertung/p_house_House_Final_0_35.npy")/(0.5 * 1.225 * 0.051985464155929494**2)


    axes[0].plot(np.linspace(0, 1, len(p[0, 2:])), np.mean(p[100000:, 2:], axis=0), label=f"$d^\\star = 60$")
    axes[1].plot(np.linspace(0, 1, len(p[0, 2:])), np.std(p[100000:, 2:], axis=0), label=f"$d^\\star = 60$")
    axes[2].plot(np.linspace(0, 1, len(p[0, 2:])), np.min(p[100000:, 2:], axis=0), label=f"$d^\\star = 60$")

    axes[0].plot(np.linspace(0, 1, len(p_large[0, 2:])), np.mean(p_large[100000:, 2:], axis=0), label=f"$d^\\star = 80$")
    axes[1].plot(np.linspace(0, 1, len(p_large[0, 2:])), np.std(p_large[100000:, 2:], axis=0), label=f"$d^\\star = 80$")
    axes[2].plot(np.linspace(0, 1, len(p_large[0, 2:])), np.min(p_large[100000:, 2:], axis=0), label=f"$d^\\star = 80$")


    for i in range(0,3):
        axes[i].legend()
        axes[i].grid()
        axes[i].set_xlabel("Position along roof surface")
        axes[i].set_ylabel(f"{lang[i]} of $C_p$")
        axes[i].set_title(f"{lang[i]}")

    tikz.clean_figure()
    tikz.save(f"house_p_sizeComp.tikz")
    plt.show()

    plt.figure()
    plt.plot(np.linspace(0, 1, len(p[0, 2:])), np.mean(p[100000:, 2:], axis=0), label=f"$d^\\star = 60$")
    plt.plot(np.linspace(0, 1, len(p_large[0, 2:])), np.mean(p_large[100000:, 2:], axis=0), label=f"$d^\\star = 80$")
    plt.legend()
    plt.grid()
    plt.xlabel("Position along roof surface")
    plt.ylabel(f"{lang[0]} of $C_p$")
    tikz.clean_figure()
    tikz.save(f"house_p_sizeComp_avg.tikz")
    plt.show()


"""
# Konvergence Study:
# braucht man nicht?
mean = np.zeros([195000])
for i in range(len(mean)):
    mean[i] = np.mean(p[100000:(105000 + i), 31], axis=0)
plt.figure()
plt.plot(mean)
"""