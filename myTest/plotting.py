import matplotlib.pyplot as plt
import numpy as np

# data source
data = np.loadtxt("/home/max/Desktop/plots/data/St_DpY_Re200.CSV", delimiter=";")
lines = plt.plot(*data)
plt.setp(lines, ls="--", lw=1, marker=".")

plt.xlabel("DpY")
plt.ylabel("$St$")
plt.xlim([0,80])
plt.grid()
plt.title("Strouhal-Zahl $St$ in Abhängigkeit der Domänenbreite in Durchmessern (DpY), für Re = 200", wrap=True)

literature = [0.2,0.192,0.196,0.202,0.195,0.201,0.192,0.191]
for lit in literature:
    plt.axhline(y=lit, color="r", ls="--", lw=0.5)
plt.savefig("/home/max/Desktop/plots/St_DpY_Re200.png")
plt.show()