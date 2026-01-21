import numpy as np
import matplotlib.pyplot as plt
import os

# Zeitachse von 0 bis 4 mit feiner Auflösung
t = np.linspace(0, 4, 1000)

# Grundfrequenz: Periode = 2 → f = 1/2
omega = 2 * np.pi * (1 / 2)

# Erste Welle: startet bei t=0
wave1 = np.cos(omega * t)

# Welle 2: t ab 1
t2 = t[t >= 1]
wave2 = -np.cos(omega * (t2 - 1))

# Welle 3: t ab 2
t3 = t[t >= 2]
wave3 = -np.cos(omega * (t3 - 2))

# Überlagerung der drei Wellen
#composite = wave1 + wave2 + wave3

# Plot
fig, ax = plt.subplots(figsize=(5, 4))
#plt.plot(t, composite, label='Signal auf Fluidknoten (Eingang)', color='black')
ax.plot(t, wave1, label='Eingangssignal vom Fluidknoten', color='black')
ax.plot(t3, wave3, '--', label='Signal nach FWBB', lw=2)
ax.plot(t2, wave2, '--', label='Signal nach HWBB', lw=2)


ax.set_xticks([0, 1, 2, 3, 4])
ax.set_yticks([-1, 0, 1])
ax.set_xlabel('Zeitschritt')
ax.set_ylabel('Signalrichtung auf Fluidknoten')
ax.grid(True)
#ax.legend()
#ax.legend(loc='upper center', bbox_to_anchor=(1, 0.5))
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25))
#ax.title('Überlagerung von Cosinuswellen')
plt.tight_layout()
#plt.tight_layout(rect=[0, 0, 1, 0.88])

### DATA I/O settings
output_base_path = "/home/mbille/lettuce/plotting_MA"
plot_batch_name = "Schema_HWBB_FWBB_OSZILLATION"
if not os.path.exists(output_base_path + "/" + plot_batch_name):
    os.makedirs(output_base_path + "/" + plot_batch_name)
plt.savefig(output_base_path + "/" + plot_batch_name + "/" + "Schema_HWBB_FWBB_OSZILLATION.png")
plt.show()