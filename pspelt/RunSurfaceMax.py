import sys
from datetime import datetime
from math import floor, sqrt

import torch
import numpy as np
import os
import trimesh
import lettuce as lt
from time import time, sleep
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from geometric_building_model import build_house_max
from helperFunctions.getIBBdata import getIBBdata
from helperFunctions.getInputData import getInputData, getHouse
from helperFunctions.logging import Logger
from obstacleFunctions import overlap_solids
from helperFunctions.plotting import Show2D, plot_intersection_info, print_results

from lettuce.boundary import InterpolatedBounceBackBoundary_occ, BounceBackBoundary
from lettuce.boundary import PartiallySaturatedBoundary
from obstacleSurface import ObstacleSurface

time0 = time()
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--default_device", default="cuda", type=str, help="")
parser.add_argument("--input", default='landscape_3D', type=str, help="")
parser.add_argument("--housename", default='house_max', type=str,
                    help="")
parser.add_argument("--inputtype", default='stl', type=str, help="")
parser.add_argument("--coll", default='kbc', choices=['kbc', 'bgk', 'bgk_reg'], type=str, help="")
parser.add_argument("--dim", default=2, type=int, choices=[2, 3], help="")
parser.add_argument("--u_init", default=2, type=int, choices=[0, 1, 2],
                    help="0: no velocity initially, 1: velocity one uniform, 2: velocity profile")
parser.add_argument("--res", default=3, type=float, help="points per meter")
parser.add_argument("--Re", default=0, type=float, help="")
parser.add_argument("--Ma", default=0.01, type=float, help="")
parser.add_argument("--depth", default=100, type=float, help="")
parser.add_argument("--minz", default=63, type=float, help="")
parser.add_argument("--maxz", default=123, type=float, help="")
parser.add_argument("--tmax", default=20, type=float, help="")
parser.add_argument("--interpolateres", default=None, type=float, help="")
parser.add_argument("--nmax", default=None, type=int, help="")
parser.add_argument("--outdir", default=os.getcwd(), type=str, help="")
parser.add_argument("--cluster", action='store_true', help="")
parser.add_argument("--withhouse", action='store_true', help="")
parser.add_argument("--debug", action='store_true', help="")
parser.add_argument("--landscapefwbb", action='store_true', help="")
parser.add_argument("--allfwbb", action='store_true', help="")
parser.add_argument("--parallel", action='store_true', help="")
parser.add_argument("--interpolatecsv", action='store_true', help="interpolate csv for fast prototyping")
parser.add_argument("--nout", default=200, type=int, help="")
parser.add_argument("--i_out_min", default=0, type=int, help="output only after i_out_min")
parser.add_argument("--nplot", default=500, type=int, help="")
parser.add_argument("--i_plot_min", default=0, type=int, help="plot only after i_plot_min")
parser.add_argument("--stepsize", default=500, type=int, help="")
parser.add_argument("--collision_data_path", default=os.path.join(os.getcwd(), 'collision_data'), type=str, help="")
parser.add_argument("--no_store_coll", action='store_true', help="")
parser.add_argument("--double_precision", action='store_true', help="")
parser.add_argument("--recalc", action='store_true', help="recalculate collision data")
parser.add_argument("--notree", action='store_true', help="")
parser.add_argument("--vmax", default=1, type=float, help="note: estimate!")  # not: estimate!
parser.add_argument("--saturation", default=0.5, type=float, help="canopy partial saturation")  # not: estimate!
parser.add_argument("--cut_z", default=0, type=float, help="cut at z=")

args = vars(parser.parse_args())
dim, coll, debug, default_device, res, outdir, u_init, nout, Ma, tmax, nmax, allfwbb, depth, cluster, withhouse, \
    no_store_coll, collision_data_path, landscapefwbb, inputtype, recalc, parallel, notree, vmax = \
    [args[_] for _ in ['dim', 'coll', 'debug', 'default_device', 'res', 'outdir', 'u_init', 'nout', 'Ma', 'tmax',
                       'nmax', 'allfwbb', 'depth', 'cluster', 'withhouse', 'no_store_coll',
                       'collision_data_path', 'landscapefwbb', 'inputtype', 'recalc', 'parallel', 'notree', 'vmax']]

#ACHTUNG! torch.set_default_device(default_device)
viscosity_pu = 14.88 * 10 ** -6  # air kinematic viscosity at 18°C from https://www.engineeringtoolbox.com/air-absolute-kinematic-viscosity-d_601.html
char_leng_pu = 1  # 1m, because
char_velo_pu = 0.42  # v_ref according to Alex  # TODO: raise v_ref to 1[LU] and adapt ux
Re = char_leng_pu * char_velo_pu / viscosity_pu  # = 2.8e5
if args["Re"] != 0:
    Re = args["Re"]

outdir = os.path.join(outdir, f"{'house' if withhouse else 'nohouse'}{dim}D_{'notree_' if notree else ''}"
                              f"_{'lsfwbb_' if landscapefwbb else ''}Re{Re:.0f}"
                              f"_Ma{Ma:.3f}_{res:.2f}ppm_{coll}_u{u_init}_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
print(f"Outdir = {outdir}")
if not os.path.exists(outdir):
    os.mkdir(outdir)
old_stdout = sys.stdout
sys.stdout = Logger(outdir)

print(f"Input: {args}")
print(f"Re: {Re:.1f}, Ma: {Ma}")

# coordinates (all pu)
xmin, ymin, zmin = 0, 0, 0
xmax, ymax, zmax = 60, 30, 40
minz_house, maxz_house = (15, 20) if dim == 3 else (-1, 1)
floor_height = 1/res * .5
house_coordinates = [[15, 0+floor_height], [15, 10+floor_height], [14, 10+floor_height], [20, 15.5+floor_height],
                     [26, 10+floor_height], [25, 10+floor_height], [25, 0+floor_height]]
domain_constraints = ([xmin, ymin], [xmax, ymax]) if dim == 2 else ([xmin, ymin, zmin], [xmax, ymax, zmax])
lx, ly, lz = xmax-xmin, ymax-ymin, zmax-zmin
shape = (int(lx*res), int(ly*res)) if dim == 2 else (int(lx*res), int(ly*res), int(lz*res))

lattice = lt.Lattice(lt.D2Q9 if dim == 2 else lt.D3Q27, device=default_device, #use_native=False,
                     dtype=torch.float64 if args["double_precision"] else torch.float)

flow = ObstacleSurface(
    res=res,
    shape=shape,
    reynolds_number=Re,
    mach_number=Ma,
    u_init=u_init,  # apply velocity profile
    lattice=lattice,
    char_length_pu=1,
    char_velocity_pu=vmax,
    domain_constraints=domain_constraints,
    depth=depth,
    debug=False,
    fwbb=args['allfwbb'],
    parallel=parallel,
    cluster=cluster
)

house_name = args['housename']
if dim == 3:
    house_name += '3d'
house_data = build_house_max(house_coordinates, minz=minz_house, maxz=maxz_house)
house_coll = getIBBdata(house_data, torch.tensor(np.array(flow.grid), device=lattice.device), lattice, no_store_coll, res, dim, house_name,
                        collision_data_path, redo_calculations=recalc, parallel=parallel, device=default_device,
                        cluster=cluster)
# plot_intersection_info(house_coll, torch.tensor(flow.grid, device=lattice.device), lattice, house_coll.solid_mask, outdir, name=house_name)

flow.add_boundary(house_data, BounceBackBoundary if args["allfwbb"] else InterpolatedBounceBackBoundary_occ,
                  collision_data=house_coll, name=house_name, ad_enabled=True)

x, y = torch.tensor(np.array(flow.grid), device=lattice.device)[0:2]
flow.add_boundary(y < floor_height, BounceBackBoundary, name="solid bottom")

if coll == 'bgk':
    collision = lt.BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
elif coll == 'bgk_reg' and dim == 2:
    collision = lt.RegularizedCollision(lattice, tau=flow.units.relaxation_parameter_lu)
elif coll == 'kbc' and dim == 2:
    collision = lt.KBCCollision2D(lattice, tau=flow.units.relaxation_parameter_lu)
elif coll == 'kbc' and dim == 3:
    collision = lt.KBCCollision3D(lattice, tau=flow.units.relaxation_parameter_lu)
else:
    raise ValueError("Invalid collision")
streaming = lt.StandardStreaming(lattice)
simulation = lt.Simulation(flow=flow, lattice=lattice, collision=collision, streaming=streaming)
simulation.reporters.append(lt.VTKReporter(lattice, flow, interval=nout, filename_base=os.path.join(outdir, "vtk"),
                                           imin=args['i_out_min']))
simulation.initialize_f_neq()
u0max = flow.ux.max().item()
show2d = Show2D(lattice, flow.solid_mask, domain_constraints, outdir=outdir, show=not cluster)
show2d(flow.units.convert_velocity_to_pu(lattice.u(simulation.f))[0], "u_x(t=0)", "u_0", vlim=(-.2, u0max))

# run simulation
steps, mlups_sum = 0, 0
step_size = args['nplot']
step_max = flow.units.convert_time_to_lu(tmax)
if nmax is not None:
    step_max = min(step_max, nmax)
print(f"Step size: {step_size}, max steps: {step_max:.1f}, max time: {tmax}")
if args["i_plot_min"] > 0:
    mlups_new = simulation.step(args["i_plot_min"])
    steps += args["i_plot_min"]
    mlups_sum += mlups_new
    mlups_avg = mlups_sum / steps
    #print_results(lattice, simulation, res, dim, flow, steps, time0, mlups_avg, show2d, u0max)
    print_results(lattice, simulation, res, dim, flow, steps, time0, mlups_new, show2d, u0max)

while steps < step_max and not torch.isnan(simulation.f).any():
    t0 = time()
    mlups_new = simulation.step(step_size)
    steps += step_size
    mlups_sum += mlups_new
    mlups_avg = mlups_sum / steps
    #print_results(lattice, simulation, res, dim, flow, steps, time0, mlups_avg, show2d, u0max)
    print_results(lattice, simulation, res, dim, flow, steps, time0, mlups_new, show2d, u0max)
    t1 = time()
    if not cluster:
        sleep(max(6.5 - (t1 - t0), 0))  # make each loop take 6.5 s to avoid "HTTP Error 429: Too Many Requests"

sys.stdout = old_stdout
