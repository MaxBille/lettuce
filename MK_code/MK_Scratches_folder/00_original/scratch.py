import numpy as np
import tikzplotlib as tikz
import matplotlib.pyplot as plt

u = np.load(f"/home/martin/Ablage/u_auswertung/u_verify_RefVel_1_3_16.7deg.npy")[20000:, ...]

print(u.shape)