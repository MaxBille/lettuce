import sys
import time
from datetime import datetime
from math import floor

import torch
import os
import matplotlib as mpl
from matplotlib import pyplot as plt
import lettuce as lt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from geometric_building_model import build_house, build_house_max
from helperFunctions.getIBBdata import getIBBdata
from helperFunctions.getInputData import getInputData, getHouse
from helperFunctions.logging import Logger
from helperFunctions.plotting import collect_intersections, plot_intersections
from obstacleFunctions import makeGrid

# import plotly.figure_factory as ff

mpl.use('TkAgg')

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--default_device", default="cuda", type=str, help="")
parser.add_argument("--dim", default=2, type=int, choices=[2, 3], help="")
parser.add_argument("--res", default=3, type=float, help="points per meter")
parser.add_argument("--depth", default=1, type=float, help="")
parser.add_argument("--outdir", default=os.getcwd(), type=str, help="")
parser.add_argument("--name", default='house_max', type=str, help="")
parser.add_argument("--debug", action='store_true', help="")
parser.add_argument("--collision_data_path", default=os.path.join(os.getcwd(), 'collision_data'), type=str, help="")
parser.add_argument("--no_store_coll", action='store_true', help="")
parser.add_argument("--recalc", action='store_true', help="recalculate collision data")
parser.add_argument("--cluster", action='store_true', help="")
parser.add_argument("--cut_z", default=0, type=float, help="cut at z=")

args = vars(parser.parse_args())
dim, res, outdir, debug, collision_data_path, default_device, no_store_coll, name, depth, recalc, cluster = \
    [args[_] for _ in ['dim', 'res', 'outdir', 'debug', 'collision_data_path', 'default_device', 'no_store_coll',
                       'name', 'depth', 'recalc', 'cluster']]

torch.set_default_device(default_device)

print(f"Input: {args}")

xmin, ymin, zmin = 0, 0, 0
xmax, ymax, zmax = 60, 30, 40
minz_house, maxz_house = (15, 20) if dim == 3 else (-1, 1)
house_coordinates = [[15, 0], [15, 10], [14, 10], [20, 15.5], [26, 10], [25, 10], [25, 0]]
floor_height = 1
domain_constraints = ([xmin, ymin], [xmax, ymax]) if dim == 2 else ([xmin, ymin, zmin], [xmax, ymax, zmax])
xmin, ymin, zmin = 0, 0, 0
xmax, ymax, zmax = 60, 30, 40
minz_house, maxz_house = (15, 20) if dim == 3 else (-1, 1)
house_coordinates = [[15, 0], [15, 10], [14, 10], [20, 15.5], [26, 10], [25, 10], [25, 0]]
floor_height = 1
domain_constraints = ([xmin, ymin], [xmax, ymax]) if dim == 2 else ([xmin, ymin, zmin], [xmax, ymax, zmax])
lx, ly, lz = xmax-xmin, ymax-ymin, zmax-zmin
shape = (int(lx*res), int(ly*res)) if dim == 2 else (int(lx*res), int(ly*res), int(lz*res))
solid = build_house_max(house_coordinates, minz=minz_house, maxz=maxz_house)


grid = makeGrid(domain_constraints, shape)
lattice = lt.Lattice(lt.D2Q9 if dim == 2 else lt.D3Q27, device=default_device, use_native=False)

cut_z = args["cut_z"]
if cut_z != 0 and dim == 2:
    name += '_' + str(cut_z)
coll_data = getIBBdata(solid, grid, lattice, no_store_coll, res, dim, name,
                       collision_data_path, redo_calculations=recalc, device=default_device, cut_z=cut_z,
                       cluster=cluster)
fluid_coords, dir_coords, surface_coords = collect_intersections(coll_data, grid, lattice)

fluid_x, fluid_y, fluid_z = fluid_coords
dir_x, dir_y, dir_z = dir_coords
surface_x, surface_y, surface_z = surface_coords

# quiver plot in pyplot
fig, ax = plt.subplots(figsize=(20, 2), dpi=600)
ax.quiver(fluid_x, fluid_y, dir_x, dir_y, color='orange', angles='xy', scale_units='xy', scale=1, label='IBB vectors')
ax.scatter(grid[0][coll_data.solid_mask].cpu(), grid[1][coll_data.solid_mask].cpu(),
           color='k', s=.5, alpha=0.4, marker='.', label='solid_mask')
ax.scatter(grid[0].cpu(), grid[1].cpu(), s=.5, alpha=0.2, marker='.', label='grid')
ax.scatter(surface_x, surface_y, s=1, marker='.', label='intersection points')
# ax.legend(fontsize='x-small')
ax.axis('equal')
plt.show()
