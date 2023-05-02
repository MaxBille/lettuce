import matplotlib.pyplot as plt
import numpy as np

# data source

data = np.loadtxt("/home/max/Desktop/plots/data/Collision_DpY50.CSV", delimiter=";")
fig, ax = plt.subplots()
p = plt.bar(["BGK", "KBC", "Regularized"],[data[1][1], data[2][1], data[3][1]])
#plt.setp(lines, ls="--", lw=1, marker=".")

plt.xlabel("Kollisionsoperator")
plt.ylabel("$C_{D}$")
plt.ylim([1.15,1.5])
plt.grid()
plt.title("Widerstandsbeiwert $C_{D}$ für verschiedene Kollisionsoperatoren, für Re = 200, GPD = 70", wrap=True)
#plt.legend(["BGK", "KBC", "Regularized"])
ax.bar_label(p, label_type="edge")

literature = [1.4,1.31,1.19,1.31,1.33,1.172,1.29,1.45,1.26,1.36,1.4087]
for lit in literature:
    plt.axhline(y=lit, color="r", ls="--", lw=0.5)
plt.savefig("/home/max/Desktop/plots/Collision_DpY50.png")
plt.show()