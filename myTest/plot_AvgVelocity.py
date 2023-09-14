import lettuce as lt
import matplotlib.pyplot as plt
import numpy as np

path_root = "/mnt/ScratchHDD1/Max_Scratch/lbm_simulations/"
sim_dir = "data_230913_155749_cyl3D_test_AvgVelocityRefPlot_GPD10_20x10x3_D3Q27/"
flow_shape = (200,100,30)
gpd=10

avg_u_start = 0.5

u1 = np.load(path_root+sim_dir+"AvgVelocity_1.npy")
u2 = np.load(path_root+sim_dir+"AvgVelocity_2.npy")
u3 = np.load(path_root+sim_dir+"AvgVelocity_3.npy")

u1 = u1[int(avg_u_start*u1.shape[0]-1):]
u2 = u2[int(avg_u_start*u2.shape[0]-1):]
u3 = u3[int(avg_u_start*u3.shape[0]-1):]

avg_u1 = np.mean(u1, axis=0)  # time average
avg_u2 = np.mean(u2, axis=0)  # time average
avg_u3 = np.mean(u3, axis=0)  # time average

avg_u1_x = avg_u1[0]  # u_x component over y at pos 1
avg_u2_x = avg_u2[0]  # u_x component over y at pos 2
avg_u3_x = avg_u3[0]  # u_x component over y at pos 3

avg_u1_y = avg_u1[1]  # u_y component over y at pos 1
avg_u2_y = avg_u2[1]  # u_y component over y at pos 2
avg_u3_y = avg_u3[1]  # u_y component over y at pos 3

y_in_D = (np.arange(avg_u1_x.shape[0])+1-flow_shape[1]/2)/gpd  # y/D for figure

# PLOT ux
fig, ax = plt.subplots(constrained_layout=True)
ax.plot(y_in_D,avg_u1_x, y_in_D, avg_u2_x, y_in_D, avg_u3_x)
ax.set_xlabel("y/D")
ax.set_ylabel(r"$\bar{u}_{x}$/$u_{char}$")
ax.legend(["x/D = 1.06", "x/D = 1.54", "x/D = 2.02"])

# OPT. TO DO: add secondary axis for LU-grid
# ...needs 'function' to convert from y/D in LU and LU in y/D

# OPT. TO DO: make folder for AvgVelocity-stuff
plt.show()

# PLOT uy
fig, ax = plt.subplots(constrained_layout=True)
ax.plot(y_in_D,avg_u1_y, y_in_D, avg_u2_y, y_in_D, avg_u3_y)
ax.set_xlabel("y/D")
ax.set_ylabel(r"$\bar{u}_{y}$/$u_{char}$")
ax.legend(["x/D = 1.06", "x/D = 1.54", "x/D = 2.02"])

# OPT. TO DO: add secondary axis for LU-grid
# ...needs 'function' to convert from y/D in LU and LU in y/D
# OPT. TO DO: make folder for AvgVelocity-stuff
# !!! QUESTION: is x/D the position measured FROM the cylinder (x_pos), or measured from x=0 ?
plt.show()

# diff between timeseries and time_average -> u'
u1_diff = u1-avg_u1
u2_diff = u2-avg_u2
u3_diff = u3-avg_u3

# square of diff -> u'^2
u1_diff_sq = u1_diff**2
u2_diff_sq = u2_diff**2
u3_diff_sq = u3_diff**2

# ux'*uy'
u1_diff_xy = u1_diff_sq[:, 0, :]*u1_diff[:, 1, :]
u2_diff_xy = u2_diff_sq[:, 0, :]*u2_diff[:, 1, :]
u3_diff_xy = u3_diff_sq[:, 0, :]*u3_diff[:, 1, :]

# time_average of u'² and ux'uy'
u1_diff_sq_mean = np.mean(u1_diff_sq, axis=0)  # time average
u2_diff_sq_mean = np.mean(u2_diff_sq, axis=0)  # time average
u3_diff_sq_mean = np.mean(u3_diff_sq, axis=0)  # time average
u1_diff_xy_mean = np.mean(u1_diff_xy, axis=0)  # time average
u2_diff_xy_mean = np.mean(u2_diff_xy, axis=0)  # time average
u3_diff_xy_mean = np.mean(u3_diff_xy, axis=0)  # time average

fig, ax = plt.subplots(constrained_layout=True)
ax.plot(y_in_D,u1_diff_sq_mean[0],y_in_D,u2_diff_sq_mean[0],y_in_D,u3_diff_sq_mean[0])
ax.set_xlabel("y/D")
ax.set_ylabel(r"$\overline{u_{x}'u_{x}'}$/$u_{char}^2$")
ax.legend(["x/D = 1.06", "x/D = 1.54", "x/D = 2.02"])
plt.show()

fig, ax = plt.subplots(constrained_layout=True)
ax.plot(y_in_D,u1_diff_sq_mean[1], y_in_D,u2_diff_sq_mean[1], y_in_D,u3_diff_sq_mean[1])
ax.set_xlabel("y/D")
ax.set_ylabel(r"$\overline{u_{y}'u_{y}'}$/$u_{char}^2$")
ax.legend(["x/D = 1.06", "x/D = 1.54", "x/D = 2.02"])
plt.show()

fig, ax = plt.subplots(constrained_layout=True)
ax.plot(y_in_D,u1_diff_xy_mean, y_in_D,u2_diff_xy_mean, y_in_D,u3_diff_xy_mean)
ax.set_xlabel("y/D")
ax.set_ylabel(r"$\overline{u_{x}'u_{y}'}$/$u_{char}^2$")
ax.legend(["x/D = 1.06", "x/D = 1.54", "x/D = 2.02"])
plt.show()