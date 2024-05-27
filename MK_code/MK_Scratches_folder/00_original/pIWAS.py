import numpy as np
import torch
from matplotlib import pyplot as plt

p30 = np.load("/mnt/hgfs/VMshare/final/p_proper_ggwPUfinal_30deg_Re20000.npy")
p45 = np.load("/mnt/hgfs/VMshare/final/p_proper_ggwPUfinal_45deg_Re20000.npy")
p60 = np.load("/mnt/hgfs/VMshare/final/p_proper_ggwPUfinal_60deg_Re20000.npy")

plt.figure()
plt.plot(p30[:, 1], p30[:, 2])
plt.plot(p45[:, 1], p45[:, 2])
plt.plot(p60[:, 1], p60[:, 2])
#N=500
#test = np.convolve(p[:, 6], np.ones(N)/N, mode='valid')
#plt.plot(p[-len(test):, 1], test)
#plt.plot(p45[:, 1], p45[:, 11])
#plt.figure()
#plt.boxplot(p30[:, 2],positions = [1], widths = 0.6)
#plt.boxplot(p45[:, 2],positions = [2], widths = 0.6)
#plt.boxplot(p60[:, 2],positions = [3], widths = 0.6)

plt.figure()
sp = np.fft.fft(p45[:, 2])
freq = np.fft.fftfreq(p45[:, 2].shape[-1], d=p45[2, 1]-p45[1, 1])
plt.plot(freq, sp.real**2 + sp.imag**2)
plt.show()
#---------------------------------
number_of_bins = 8000

# An example of three data sets to compare
number_of_data_points = 387
labels = ["A", "B", "C"]
data_sets = [p30[:, 2], p45[:, 2], p60[:, 2]]

# Computed quantities to aid plotting
hist_range = (np.min(data_sets), np.max(data_sets))
binned_data_sets = [
  np.histogram(d, range=hist_range, bins=number_of_bins)[0]
  for d in data_sets
]
binned_maximums = np.max(binned_data_sets, axis=1)
x_locations = np.arange(0, sum(binned_maximums), np.max(binned_maximums))

# The bin_edges are the same for all of the histograms
bin_edges = np.linspace(hist_range[0], hist_range[1], number_of_bins + 1)
centers = 0.5 * (bin_edges + np.roll(bin_edges, 1))[:-1]
heights = np.diff(bin_edges)

# Cycle through and plot each histogram
fig, ax = plt.subplots()
for x_loc, binned_data in zip(x_locations, binned_data_sets):
  lefts = x_loc - 0.5 * binned_data
  ax.barh(centers, binned_data, height=heights, left=lefts)

ax.set_xticks(x_locations)
ax.set_xticklabels(labels)

ax.set_ylabel("Data values")
ax.set_xlabel("Data sets")
del p30, p45, p60
plt.show()
