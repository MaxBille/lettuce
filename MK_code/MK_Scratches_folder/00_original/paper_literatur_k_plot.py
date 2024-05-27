import numpy as np
from matplotlib import pyplot as plt
import tikzplotlib as tikz


diffz = []
txt = [167, 266, 369]
txt3 = ["16.7", "26.6", "36.9"]
txt2 = [3, 5, 7]
angle = [a / 180 * np.pi for a in [16.7, 26.6, 36.9]]
offset = [-1, -0.55, 0, 0.55, 1]
z = np.linspace(0, (179/(60/1.1)), 180)

u_0u = 0.051985464155929494

def k(u):
    return 0.5 * (np.std(u[:, 0, ...], axis=0)**2 + np.std(u[:, 1, ...], axis=0)**2 + np.std(u[:, 2, ...], axis=0)**2)/u_0u**2

def calculate_errors_k():
    rel_error_ref = np.zeros(3)
    rel_error_range = np.zeros(3)
    abs_error = np.zeros(3)
    corr = np.zeros(3)
    min = 10
    max = 0

    for a in range(0, 3):
        if a > 0:
            del u
        if a == 1:
            u = np.load(f"/home/martin/Ablage/u_auswertung/u_verify_RefVel_0.8u0_1_3_26.6deg.npy")[100000:, ...]
            print("switch")
        else:
            u = np.load(f"/home/martin/Ablage/u_auswertung/u_verify_RefVel_1_3_{txt3[a]}deg.npy")[100000:, ...]
        u = k(u[:, :, :, 1:])
        if np.min(u) < min:
            min = np.min(u)
        if np.max(u) > max:
            max = np.max(u)
        rel_error_ref_l = []
        rel_error_range_l = []
        abs_error_l = []
        corr_l = []
        offset = [1, 0.55, 0, -0.55, -1]
        for i in range(1, 6):
            ref = np.genfromtxt(f'/home/martin/Ablage/p_auswertung/k_{2 * a + 3}10_{i}.csv', delimiter=',')
            ref[:, 0] = (-ref[:, 0] + offset[i - 1]) * 0.4
            ref = ref[::-1, :]
            values = []
            for j in range(len(ref[:, 0])):
                pos = np.where((np.abs(z - ref[j, 1]) == np.min(np.abs(z - ref[j, 1]))))
                values.append(u[i][pos[0]])
                abs_error_l.append(np.abs((values[j] - ref[j, 0])))
                rel_error_ref_l.append(np.abs((values[j] - ref[j, 0])/ref[j, 0]))
                rel_error_range_l.append(np.abs((values[j] - ref[j, 0])))
            corr_l.append(np.corrcoef(np.array(values).squeeze(), ref[:, 0])[0, 1])
        corr[a] = np.mean(np.array(corr_l))
        abs_error[a] = np.mean(np.array(abs_error_l))
        rel_error_range[a] = np.mean(np.array(rel_error_range_l))
        rel_error_ref[a] = np.mean(np.array(rel_error_ref_l))
    val_range = max - min
    for i in range(0,3):
        print(f"Error results: {txt3[i]}°: abs: {abs_error[i]}, rel_ref: {rel_error_ref[i]}, rel_range: {rel_error_range[i]/val_range}, range: {val_range}, correlation : {corr[i]}")

calculate_errors_k()

for h in range(0, 3):
    ref = []
    if h > 0:
        del u1
        del text, lines, plots, f
    u1 = np.load(f"/home/martin/Ablage/u_auswertung/u_verify_RefVel_1.75_2_{txt3[h]}deg.npy")[100000:, ...]
    for i in range(1, 6):
        ref.append(np.genfromtxt(f'/home/martin/Ablage/p_auswertung/k_{txt2[h]}10_{i}.csv', delimiter=','))
    for i in range(0, 5):
        plt.plot(k(u1[:, :, i + 1, 1:]) / 0.2 * 0.5 + offset[i], z[1:], color="black")
        plt.plot(-ref[i][:, 0], ref[i][:, 1], marker='o', color="gray", linestyle="None")
    plt.grid()
    plt.ylabel("$z/z_{\mathrm{roof}}$")
    plt.xlabel("$x/z_{\mathrm{roof}}$")
    plt.xlim(-1.5, 1.5)
    plt.ylim(0, 2.5)
    plt.title(f"{[16.7, 26.6, 36.9][h]}° roof pitch")
    tikz.clean_figure()
    #tikz.save(f"literatur_k_{int(txt[h]/10)}.tikz")

    text = tikz.get_tikz_code()  # f"literatur_u_{int(txt[h]/10)}.tikz")

    # scale und Haus einbauen
    lines = text.split("\n")
    lines[lines.index("]") - 1] += ","
    lines.insert(lines.index("]"), "clip mode=individual")
    lines.insert(lines.index("]") + 1,
                 "\draw[draw=white] (1, 2.85) -- (1.25, 2.85) node[anchor=south] {$k/u_0^2$};\n" + \
                 "\draw[line width=0.4mm] (1.5, 2.65) -- (1, 2.65);\n" + \
                 "\draw[line width=0.4mm] (1.5, 2.65) node[anchor=south] {\small 0.2} -- (1.5, 2.56);\n" + \
                 "\draw[line width=0.4mm] (1.25, 2.65) node[anchor=south] {\small 0.1} -- (1.25, 2.56);\n" + \
                 "\draw[line width=0.4mm] (1, 2.65) node[anchor=south] {\small 0} -- (1, 2.56);\n" + \
                 "\draw[dashed, line width=0.4mm] (-1, 0) -- (-1, 2.5);\n" + \
                 "\draw[dashed, line width=0.4mm] (-0.55, 1) -- (-0.55, 2.5);\n" + \
                 f"\draw[dashed, line width=0.4mm] (0, {np.tan(angle[h]) * 0.55 + 1}) -- (0, 2.5);\n" + \
                 "\draw[dashed, line width=0.4mm] (0.55, 1) -- (0.55, 2.5);\n" + \
                 "\draw[dashed, line width=0.4mm] (1, 0) -- (1, 2.5);\n" + \
                 "\\addplot [line width=0.5mm, black]\n" + \
                 "table {%\n" + \
                 "-0.55 0\n" + \
                 "-0.55 1\n" + \
                 f"0 {np.tan(angle[h]) * 0.55 + 1}\n" + \
                 "0.55 1\n" + \
                 "0.55 0\n" + \
                 "};")
    end = lines.pop(-2) + "\n"
    end = lines.pop(-3) + "\n" + end
    text = "".join([text + "\n" for text in lines])
    text = text.replace("mark size=3", "mark size=2")
    plots = text.split("\\addplot")

    for i in (4, 6, 8):
        if i == 4 or i == 8:
            min = 1
        else:
            min = np.tan(angle[h]) * 0.55 + 1
        plots[i] = "".join([text + "\n" for text in plots[i].split("\n")[:2] + [d for (d, remove) in zip(plots[i].split("\n")[2:-2], [blub > min for blub in [float(blub[1]) for blub in [bla.split(" ") for bla in plots[i].split("\n")[2:-2]]]]) if remove] + plots[i].split("\n")[-2:]])

    for i in range(2, len(plots) - 1, 2):
        plots[i], plots[i + 1] = plots[i + 1], plots[i]
    for i in range(2, len(plots)):
        plots[i] = plots[i].replace("black", "line width=0.4, black")
    text = "".join(list(plots[0]) + ["\\addplot" + text for text in plots[1:]])
    text += end
    plt.show()
    with open(f"literatur_k_{int(txt[h] / 10)}.tikz", "w") as f:
        f.write(text)