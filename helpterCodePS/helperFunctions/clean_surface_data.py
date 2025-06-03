"""
Reads a csv of unsorted 2D points, finds a first point with minimum x-coordinate and then consecutively adds the next
point with minimal distance to the list, until all points from the input array are sorted based on distance from the
previous point *(e.g. we are connecting the points in the input array to a polygonal line)*. In addition, this script
removes duplicates and writes the sorted and cleaned points back to a csv file.
"""
from time import time

import numpy as np
import torch
from math import floor

from matplotlib import pyplot as plt
from scipy.interpolate import interp1d, interp2d, griddata, Rbf


def get_clean_surface_data(csv_path: str = 'nsg_study_landscape_3D.csv', delimiter: str = ',', res: float = 1,
                           debug: bool = False, dim: int = None, interpolate: bool = False, depth=10.,
                           res_interp: float = None) -> np.array:
    # sort surface_data wrt to x (Actually I just need the starting point with min x)
    array_from_csv = np.genfromtxt(csv_path, delimiter=delimiter)
    n_points_input = array_from_csv.shape[0]
    dim_data = array_from_csv.shape[1]
    if dim is None:
        dim = dim_data
    if dim not in [2, 3] or dim_data not in [2, 3]:
        raise ValueError('Dimensions must be 2 or 3')
    if dim > dim_data:
        print('WARNING: Dimension demanded is higher than the dimension of point data. Stretching along z.')
    elif dim == 2 and dim_data == 3:
        array_from_csv = array_from_csv[:, [0, 2]]  # reduce input data to x and y coordinates
    elif dim_data == 3:
        array_from_csv = array_from_csv[:, [0, 2, 1]]  # putting y data to 2nd column
    surface_data = torch.tensor(array_from_csv)  # unique sorts per default
    lx = surface_data[:, 0].max() - surface_data[:, 0].min()
    ly = surface_data[:, 1].max() - surface_data[:, 1].min()
    lz = surface_data[:, 2].max() - surface_data[:, 2].min() if dim_data == 3 else depth if dim == 3 else 0
    # z is interpreted as being the cross-stream axis. In the csv data, this is the 3rd column
    unique_coordinates, inverse_indices, counts = torch.unique(surface_data[:, [0, 2] if dim_data == 3 else 0], dim=0,
                                                               return_counts=True, return_inverse=True)
    n_interpolated = 0
    if len(counts) < len(surface_data):
        print("WARNING: Some coordinates were duplicates. They may contain vertical information. "
              "Currently, their max is used.")
        unique_ys = torch.zeros(len(counts))
        if dim_data == 3:
            for i in np.arange(len(unique_coordinates)):
                x, z = unique_coordinates[i]
                ys = surface_data[torch.where((x == surface_data[:, 0]) * (z == surface_data[:, 2]))][:, 1]
                unique_ys[i] = ys.max()  # TODO: implement interpolation for duplicates
            sorted_pnts = torch.stack((unique_coordinates[:, 0], unique_ys[:], unique_coordinates[:, 1]), dim=1)
        else:
            for i in np.arange(len(unique_coordinates)):
                x = unique_coordinates[i]
                ys = surface_data[torch.where(x == surface_data[:, 0])][:, 1]
                unique_ys[i] = ys.max()  # TODO: implement interpolation for duplicates
            sorted_pnts = torch.stack((unique_coordinates[:], unique_ys[:]), dim=1)
    else:
        sorted_pnts = surface_data
    if interpolate:  # interpolate csv for fast prototyping
        res_interp = 0.5*res if res_interp is None else res_interp
        print(f"interpolating csv for {res_interp:.1f} points per meter for fast prototyping")
        time0 = time()
        x_data = sorted_pnts[:, 0].cpu()
        y_data = sorted_pnts[:, 1].cpu()
        nx = int(lx * res_interp)  # interpolate around two points
        if dim == 2 or dim_data == 2:
            x_interp = np.linspace(x_data.min(), x_data.max(), nx)  # [xmin ... xmax]
            y_interp = interp1d(x_data, y_data, kind='cubic')(x_interp)
            sorted_pnts = torch.tensor([[x_interp[_], y_interp[_]] for _ in range(nx)])
            if dim == 3:
                sorted_pnts = torch.stack((sorted_pnts[:, 0], sorted_pnts[:, 1],
                                           torch.zeros_like(sorted_pnts[:, 0])), dim=1)
        else:
            z_data = sorted_pnts[:, 2].cpu()
            xz_data = sorted_pnts[:, [0, 2]].cpu()
            nz = int(nx / lx * lz)
            x_interp = np.linspace(x_data.min(), x_data.max(), nx)  # [xmin ... xmax]
            z_interp = np.linspace(z_data.min(), z_data.max(), nz)
            # TODO: fix 2d interpolation (currently: 'OverflowError: Too many data points to interpolate')
            grid_x, grid_z = np.meshgrid(x_interp, z_interp)
            # y_interp = interp2d(xz_data, y_data, (grid_x, grid_z))
            if len(x_data) * len(y_data) * len(z_data) < 1e10:
                y_interp = Rbf(x_data, z_data, y_data, function="multiquadric", smooth=5)(grid_x, grid_z)
                sorted_pnts = torch.tensor([[grid_x[ix, iz], y_interp[ix, iz], grid_z[ix, iz]]
                                            for ix in range(nx) for iz in range(nz)])
            else:
                y_interp = griddata(xz_data, y_data, (grid_x, grid_z)).transpose()
                sorted_pnts = torch.tensor([[grid_x.transpose()[ix, iz], y_interp[ix, iz], grid_z.transpose()[ix, iz]]
                                            for iz in np.arange(1, nz - 1)
                                            for ix in np.arange(1, nx - 1)])
        time1 = time() - time0
        # plotting interpolated grid
        fig, ax = plt.subplots()
        ax.scatter(sorted_pnts[:, 0].cpu(), sorted_pnts[:, 1].cpu())
        plt.show()
        plt.close()
        print(f"Interpolating csv data took {floor(time1 / 60):02d}:{floor(time1 % 60):02d} [mm:ss].")
    print(f"Points in input: {n_points_input}. Points kept: {len(sorted_pnts)}. "
          f"Duplicates: {n_points_input - len(sorted_pnts)} Interpolated points: {n_interpolated}")
    return sorted_pnts, dim, dim_data
