import numpy as np
from tikzplotlib import save
import matplotlib
#matplotlib.use("pgf")
import matplotlib.pyplot as plt
from matplotlib import cm

if 0:
    for i, angle in enumerate([20, 35, 50]):
        p = np.load(f"/home/martin/Masterthesis/Clusterinhalt/final_finally/p_final2_noneQfinal_noverhang_{angle}.0deg_Re20000.npy")
        if i == 0:
            mean = np.zeros([3, p.shape[1]])
            max = np.zeros([3, p.shape[1]])
            min = np.zeros([3, p.shape[1]])
            std = np.zeros([3, p.shape[1]])
        for j in range(2,p.shape[1]):
            if angle != 20:
                end = len(p[:, j])
            else:
                end = 250000
            mean[i, j] = np.mean(p[100000:end, j]/(0.5 * 1.225 * 0.029705979517673998**2))
            max[i, j] = np.max(p[100000:end, j]/(0.5 * 1.225 * 0.029705979517673998**2))
            min[i, j] = np.min(p[100000:end, j]/(0.5 * 1.225 * 0.029705979517673998**2))
            std[i, j] = np.std(p[100000:end, j]/(0.5 * 1.225 * 0.029705979517673998**2))
        del p
    np.save("/home/martin/lettuce/p_nover_mean.npy", mean)
    np.save("/home/martin/lettuce/p_nover_max.npy", max)
    np.save("/home/martin/lettuce/p_nover_min.npy", min)
    np.save("/home/martin/lettuce/p_nover_std.npy", std)
else:
    mean = np.load("/home/martin/lettuce/p_nover_mean.npy")
    max = np.load("/home/martin/lettuce/p_nover_max.npy")
    min = np.load("/home/martin/lettuce/p_nover_min.npy")
    std = np.load("/home/martin/lettuce/p_nover_std.npy")

    mean2 = np.load("/home/martin/lettuce/p_mean.npy")
    max2 = np.load("/home/martin/lettuce/p_max.npy")
    min2 = np.load("/home/martin/lettuce/p_min.npy")
    std2 = np.load("/home/martin/lettuce/p_std.npy")

    mean = np.vstack([mean2, mean])
    max = np.vstack([max2, max])
    min = np.vstack([min2, min])
    std = np.vstack([std2, std])





bla = []
for i, angle in enumerate([20, 35, 50]):
    posz = np.load(f"/home/martin/Masterthesis/Clusterinhalt/final_finally/p{angle}.npy")
    bla.append(posz)
for i, angle in enumerate([20, 35, 50]):
    posz = np.load(f"/home/martin/Masterthesis/Clusterinhalt/final_finally/p{angle}_nover.npy")
    bla.append(posz)
pos = np.stack((bla[0], bla[1], bla[2], bla[3], bla[4], bla[5]), axis=2)



start = 396 #264
ende = start + 132


maxx = -1
minn = 1
array = max
for file in range(0, 3):
    color_dimension = array[file, start:ende].reshape(11, 12)
    minn, maxx = np.min([color_dimension.min(), minn]), np.max([color_dimension.max(), maxx])
norm = matplotlib.colors.Normalize(minn, maxx)
m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
m.set_array([])
fcolors = []
for file in range(0, 6):
    color_dimension = array[file, start:ende].reshape(11, 12)
    fcolors.append(m.to_rgba(color_dimension))
fig = plt.figure()
ax = []
for file in range(0, 6):
    ax.append(fig.add_subplot(2, 3, file+1, projection='3d'))
    ax[-1].plot_surface(pos[start:ende, 0, file].reshape(11, 12), pos[start:ende, 1, file].reshape(11, 12),
                    pos[start:ende, 2, file].reshape(11, 12), rstride=1, cstride=1, facecolors=fcolors[file], vmin=minn,
                    vmax=maxx, shade=False)
    ax[-1].view_init(elev=60., azim=-90)
    plt.axis('off')
    plt.margins(0.0)

fig.colorbar(m, ax=ax, location="right", shrink=0.35)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())

plt.show()
#plt.savefig('/mnt/hgfs/VMshare/p_std_nover.pgf', backend='pgf', bbox_inches='tight', pad_inches=0)