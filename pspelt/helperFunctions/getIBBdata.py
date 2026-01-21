import os
from typing import Tuple

import torch
import numpy as np
import trimesh
from OCC.Core.TopoDS import TopoDS_Shape

from pspelt.helperFunctions.plotting import plot_intersection_info, Show2D
from pspelt.obstacleFunctions import overlap_solids, calculate_mask, collect_solid_boundary_data
from lettuce import Lattice
from lettuce.boundary import SolidBoundaryData


def getIBBdata(cad_data: trimesh.Trimesh or TopoDS_Shape, grid: tuple[np.ndarray, ...], periodicity: tuple[bool, ...], lattice: Lattice,
               no_store_solid_boundary_data: bool, res: float, dim: int, name: str, solid_boundary_data_path: str,
               only_mask: bool = False, redo_calculations: bool = False, parallel: bool = False,
               device: str = "cuda", cut_z: float = 0., cluster: bool = False, verbose: bool = False) -> SolidBoundaryData:
    if not os.path.exists(solid_boundary_data_path):
        os.makedirs(solid_boundary_data_path)

    solid_boundary_data_path = os.path.join(solid_boundary_data_path, f"{res:.2f}ppm_{dim}D")
    print(f"Looking for data for {name} in {solid_boundary_data_path}.")

    solid_boundary_data = SolidBoundaryData()

    mask_data_exists = (os.path.exists(os.path.join(solid_boundary_data_path, f"solid_mask_{name}.npy")) and
                        os.path.exists(os.path.join(solid_boundary_data_path, f"points_inside_{name}.npy")))
    if mask_data_exists and not redo_calculations:
        print("Mask data found.")
        solid_boundary_data.solid_mask = np.load(os.path.join(solid_boundary_data_path, f"solid_mask_{name}.npy"))
        solid_boundary_data.points_inside = np.load(os.path.join(solid_boundary_data_path, f"points_inside_{name}.npy"))
    else:
        if not os.path.exists(solid_boundary_data_path):
            os.mkdir(solid_boundary_data_path)
        print("No mask data found or recalculation requested. Redoing mask calculations.")
        solid_boundary_data = calculate_mask(cad_data, grid, name=name, solid_boundary_data=solid_boundary_data, cut_z=cut_z)
        if not no_store_solid_boundary_data:
            np.save(os.path.join(solid_boundary_data_path, f"solid_mask_{name}.npy"), solid_boundary_data.solid_mask)
            np.save(os.path.join(solid_boundary_data_path, f"points_inside_{name}.npy"), solid_boundary_data.points_inside)
            print(f"Mask data saved to {solid_boundary_data_path}.")
    print(f"Mask data loaded for {name}.")

    if only_mask:
        return solid_boundary_data
    solid_boundary_data_exists = True
    for data_name in ['f_index_gt_', 'f_index_lt_', 'd_gt_', 'd_lt_', 'not_intersected_']:
        solid_boundary_data_exists *= os.path.exists(os.path.join(solid_boundary_data_path, f"{data_name}{name}.npy"))
    if solid_boundary_data_exists and not redo_calculations:
        print("Solid Boundary Data data found.")
        solid_boundary_data.f_index_gt = np.load(os.path.join(solid_boundary_data_path, f"f_index_gt_{name}.npy"))
        solid_boundary_data.f_index_lt = np.load(os.path.join(solid_boundary_data_path, f"f_index_lt_{name}.npy"))
        solid_boundary_data.d_gt = np.load(os.path.join(solid_boundary_data_path, f"d_gt_{name}.npy"))
        solid_boundary_data.d_lt = np.load(os.path.join(solid_boundary_data_path, f"d_lt_{name}.npy"))
        solid_boundary_data.not_intersected = np.load(os.path.join(solid_boundary_data_path, f"not_intersected_{name}.npy"))
    else:
        print("No Solid Boundary Data found or recalculation requested. Redoing Solid Boundary Data calculations.")
        solid_boundary_data = collect_solid_boundary_data(cad_data, solid_boundary_data, lattice, grid, periodicity, name, outdir=solid_boundary_data_path, cut_z=cut_z, cluster=cluster, verbose=verbose)
        if not no_store_solid_boundary_data:
            np.save(os.path.join(solid_boundary_data_path, f"f_index_gt_{name}.npy"), solid_boundary_data.f_index_gt)
            np.save(os.path.join(solid_boundary_data_path, f"f_index_lt_{name}.npy"), solid_boundary_data.f_index_lt)
            np.save(os.path.join(solid_boundary_data_path, f"d_gt_{name}.npy"), solid_boundary_data.d_gt)
            np.save(os.path.join(solid_boundary_data_path, f"d_lt_{name}.npy"), solid_boundary_data.d_lt)
            np.save(os.path.join(solid_boundary_data_path, f"not_intersected_{name}.npy"), solid_boundary_data.not_intersected)
            print(f"Solid Boundary Data saved to {solid_boundary_data_path}.")
    print(f"Solid Boundary Data loaded for {name}.")

    return solid_boundary_data
