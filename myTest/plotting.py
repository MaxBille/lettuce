import matplotlib.pyplot as plt
import numpy as np

# data source
data = np.loadtxt("/home/max/Desktop/plots/data/Collision_DpY19.CSV", delimiter=";")
lines = plt.plot(data[0], data[1], data[0], data[2], data[0], data[3])
plt.setp(lines, ls="--", lw=1, marker=".")

plt.xlabel("GPD")
plt.ylabel("$C_{D}$")
#plt.xlim([0,50])
plt.grid()
plt.title("Widerstandsbeiwert $C_{D}$ in Abhängigkeit der Auflösung in GPD für verschiedene Kollisionsoperatoren, für Re = 200", wrap=True)
plt.legend(["BGK", "KBC", "Regularized"])

literature = [1.4,1.31,1.19,1.31,1.33,1.172,1.29,1.45,1.26,1.36,1.4087]
for lit in literature:
    plt.axhline(y=lit, color="r", ls="--", lw=0.5)
plt.savefig("/home/max/Desktop/plots/Collision_DpY19.png")
plt.show()