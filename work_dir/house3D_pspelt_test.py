### IMPORT

# sys, os etc.
import sys
import os
import psutil
import shutil
from time import time, sleep
import datetime
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# torch, np,
import numpy as np
import torch
from OCC.Core.TopoDS import TopoDS_Shape
from matplotlib import pyplot as plt

# lettuce
import lettuce as lt
from lettuce import torch_gradient
from lettuce.boundary import InterpolatedBounceBackBoundary_occ, BounceBackBoundary
from lettuce.boundary_mk import EquilibriumExtrapolationOutlet, NonEquilibriumExtrapolationInletU, ZeroGradientOutlet, SyntheticEddyInlet
# flow
from pspelt.obstacleSurface import ObstacleSurface
from lettuce.flows.houseFlow import HouseFlow3D

# pspelt
from pspelt.geometric_building_model import build_house_max
from pspelt.helperFunctions.getIBBdata import getIBBdata
from pspelt.helperFunctions.getInputData import getInputData, getHouse
from pspelt.helperFunctions.logging import Logger
from pspelt.obstacleFunctions import overlap_solids, makeGrid
from pspelt.helperFunctions.plotting import Show2D, plot_intersection_info, print_results
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse


### ARGUMENTS

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

# (!) action = 'store_false' bedeutet, dass im Fall des GEGEBENEN Arguments, false gespeichert wird, und wenn es NICHT gegeben ist, True... wtf

# Infrastructure and I/O (path, devices, name, walltime, vtk-output,...)
parser.add_argument("--name", default="3Dhouse", help="name of the simulation, appears in output directory name")
parser.add_argument("--default_device", default="cuda", type=str, help="run on cuda or cpu")
parser.add_argument("--float_dtype", default="foat32", choices=["foat32", "foat64", "single", "double", "half"], help="data type for floating point calculations in torch")
parser.add_argument("--t_sim_max", default=(72*60*60), type=float, help="max. walltime to simulate, default is 72 h for cluster use. sim stops at 0.99*t_max_sim")  # andere max.Zeit? wie lange braucht das "drum rum"? kann cih simulation auch die scho vergangene Zeit übergeben? dann kann ich mit nem größeren Wert rechnen und sim ist variabel darin, wie viel Zeit es noch hat

parser.add_argument("--cluster", action='store_true', help="")  # brauche ich das?
parser.add_argument("--outdir", default=os.getcwd(), type=str, help="directory to save output files to; vtk-files will be saved in seperate dir, if outputdir_vtk is specified")
parser.add_argument("--outdir_vtk", default=None, type=str, help="")
parser.add_argument("--vtk", action='store_true', help="toggle vtk-output to outdor_vtk, if set True (1)")
parser.add_argument("--vtk_fps", default=10, help="frames per second_PU for VTK output; overwritten if vtk_interval is specified")
parser.add_argument("--vtk_interval", default=0, type=int, help="how many steps between vtk-output-files; overwrites vtk_fps")
parser.add_argument("--vtk_start", default=0, type=float, help="at which percentage of t_target or n_steps the vtk-reporter starts, default is 0 (from the beginning); values from 0.0 to 1.0")

parser.add_argument("--nan_reporter", action='store_true', help="stop simulation if NaN is detected in f field")
parser.add_argument("--from_cpt", action='store_true', help="start from checkpoint. (!) provide --cpt_file path")
parser.add_argument("--cpt_file", default=None, help="path and name of cpt_file to use if --from_cpt=True")
parser.add_argument("--sim_i", default=0, type=int, help="step index of last checkpoints-step to start from for time-indexing of observables etc.")
parser.add_argument("--write_cpt", action='store_true', help="write checkpoint after finishing simulation")

# flow physics
parser.add_argument("--re", default=200, type=float, help="Reynolds number")
parser.add_argument("--ma", default=0.05, type=float, help="Mach number")
parser.add_argument("--viscosity_pu", default=14.852989758837 * 10**(-6), type=float, help="kinematic fluid viscosity in PU. Default is air at ~14.853e-6 (at 15°C, 1atm)")
parser.add_argument("--char_density_pu", default=1.2250, help="density, default is air at ~1.2250 at 15°C, 1atm")  # ist das so korrekt? - von Martin Kiemank übernommen
parser.add_argument("--u_init", default=0, type=int, choices=[0, 1, 2], help="0: initial velocity zero, 1: velocity one uniform, 2: velocity profile") # könnte ich noch auf Philipp und mich anpassen...und uniform durch komplett WSP ersetzen
# char velocity PU will be calculated from Re, viscosity and char_length!

# solver settings
parser.add_argument("--n_steps", default=100000, type=int, help="number of steps to simulate, overwritten by t_target, if t_target is >0, end of sim will be step_start+n_steps")
parser.add_argument("--t_target", default=0, type=float, help="time in PU to simulate, t_start will be calculated by PU/LU-conversion of step_start")
parser.add_argument("--step_start", default=0, type=int, help="stepnumber to start at. Useful if sim. is started from a checkpoint and sim-data should be concatenated later on")
parser.add_argument("--collision", default="bgk", choices=['kbc', 'bgk', 'bgk_reg'], help="collision operator (bgk, kbc, reg)")
parser.add_argument("--dim", default=3, type=int, help="dimensions: 2D (2), oder 3D (3, default)")
parser.add_argument("--stencil", default="D3Q27", choices=['D2Q9', 'D3Q15', 'D3Q19', 'D3Q27'], help="stencil (D2Q9, D3Q27, D3Q19, D3Q15), dimensions will be infered from D")
parser.add_argument("--eqlm", action="store_true", help="use Equilibium LessMemory to save ~20% on GPU VRAM, sacrificing ~2% performance")

# house and domain geometry
parser.add_argument("--house_length_lu", default=10, type=int, help="house length in LU")  # characteristic length LU, in flow direction
parser.add_argument("--house_length_pu", default=10, help="house length in PU")  # characteristic length PU [m]
parser.add_argument("--house_width_pu", default=0, help="width of house in crossstream direction. If left default, it will be equal to house_length_pu")  # cross-stream house_width PU [m]
#house_position  # center of house foundation (corner closest to domain origin?) / erstmal hardcoded, denn man kann als argument wohl kein tupel übergeben
parser.add_argument("--roof_angle", default=45, help="roof_angle in degree (0 to <90)")  # angle of roof (incline and decline for symmetric roof) - depending on how the house-polygon is defined, obsolete?
parser.add_argument("--eg_height_pu", default=0, help="eg_height in PU")  # if left 0, roof_height is taken. If roof_height is zero as well, eg_height_pu = house_length_pu
parser.add_argument("--roof_height_pu", default=0, help="roof_height in PU") # if left 0, roof_height is infered from eg_height and roof_angle
parser.add_argument("--overhang_pu", default=0, help="roof overhang in PU")
parser.add_argument("--domain_length_pu", default=60, type=float, help="flow-direction domain length in PU")
parser.add_argument("--domain_width_pu", default=40, type=float, help="cross-flow-direction domain width in PU")
parser.add_argument("--domain_height_pu", default=30, type=float, help="cross-flow domain height in PU")

# boundary algorithms
parser.add_argument("--inlet_bc", default="eqin", help="inlet boundary condition: EQin, NEX, SEI")
parser.add_argument("--outlet_bc", default="eqoutp", help="outlet boundary condition: EQoutP")
parser.add_argument("--ground_bc", default="fwbb", help="ground boundary condition: fwbb, hwbb")
parser.add_argument("--house_bc", default="fwbb", help="house boundary condition: fwbb, hwbb, ibb")
parser.add_argument("--top_bc", default="zgo", help="top boundary condition: zgo, eq")

parser.add_argument("--combine_solids", action='store_true', help="combine all solids (house and ground) into one object for easier prototyping")
parser.add_argument("--plot_intersection_info", action='store_true', help="plot intersection info to outdir to debug solid-boundary problems")


#pspelt
# parser.add_argument("--input", default='landscape_3D', type=str, help="") # NOT USED
# parser.add_argument("--inputtype", default='stl', type=str, help="")
# parser.add_argument("--dim", default=2, type=int, choices=[2, 3], help="")  # bei mir vom stencil abhängig
# parser.add_argument("--res", default=3, type=float, help="points per meter") # bei mir von house_length_lu/PU abhängig!
# parser.add_argument("--depth", default=100, type=float, help="") # not used? wird an ObstacleSurface übergeben, aber nicht genutzt...
# parser.add_argument("--minz", default=63, type=float, help="")  # IST DAS LU oder PU? warhscheinlich LU...aber das ist irgendwie inkonsequent
# parser.add_argument("--maxz", default=123, type=float, help="")
# parser.add_argument("--interpolateres", default=None, type=float, help="") # NOT USED
# parser.add_argument("--nmax", default=None, type=int, help="") # bei mir "n_steps
# parser.add_argument("--withhouse", action='store_true', help="")  # bei mir nicht gebraucht
# parser.add_argument("--debug", action='store_true', help="")  # wird nur an obstacleSurface übergeben, aber nicht genutzt
# parser.add_argument("--landscapefwbb", action='store_true', help="") # NOT USED
# parser.add_argument("--allfwbb", action='store_true', help="")  # da könnte ich was ähnliche machen wie boundary_combine... aber wie gebe ich dann an welche? FWBB/HWBB/IBB genutzt werden soll
# parser.add_argument("--parallel", action='store_true', help="") # parallelisierung der Nachbarsuche, die aber nich so richtig zu funktionieren scheint. Vielleicht kann man das auf CPU parallelisieren?
# parser.add_argument("--interpolatecsv", action='store_true', help="interpolate csv for fast prototyping") # NOT USED
# parser.add_argument("--nout", default=200, type=int, help="") # bei mir vtk_fps bzw. vtk_interval
# parser.add_argument("--i_out_min", default=0, type=int, help="output only after i_out_min") # bei mir als vtk_start drin, aber relativ
# parser.add_argument("--nplot", default=500, type=int, help="") # alle wie viele Schritte geplottet wird bei Philipp
# parser.add_argument("--i_plot_min", default=0, type=int, help="plot only after i_plot_min") # ab wann geplottet wird, bei Philipp
# parser.add_argument("--stepsize", default=500, type=int, help="") # NOT USED
parser.add_argument("--collision_data_path", default=os.path.join(os.getcwd(), 'collision_data'), type=str, help="")  # DAS BRAUCH ICH...
parser.add_argument("--no_store_coll", action='store_true', help="") # ob coll_data gespeichert wird, oder nicht... -> ohne, wirds zwar verwendet, aber nicht gespeichert
# parser.add_argument("--double_precision", action='store_true', help="") # ist bei mir als float_dtype hinterlegt (s.o.)
parser.add_argument("--recalc", action='store_true', help="recalculate collision data") # DAS BRAUCHE ICH AUCH
# parser.add_argument("--notree", action='store_true', help="") # NOT USED
# parser.add_argument("--vmax", default=1, type=float, help="note: estimate!")  # not: estimate! / das ist char_velocity PU
# parser.add_argument("--saturation", default=0.5, type=float, help="canopy partial saturation")  # not: estimate! / NOT USED
# parser.add_argument("--cut_z", default=0, type=float, help="cut at z=") # NOT USED

args = vars(parser.parse_args())

# get parameters from args[] dict:
name, default_device, float_dtype, t_sim_max, cluster, outdir, outdir_vtk, vtk, vtk_fps, vtk_interval, vtk_start, \
    nan_reporter, from_cpt, sim_i, write_cpt, re, ma, viscosity_pu, char_density_pu, u_init, n_steps, t_target, \
    step_start, collision, dim, stencil, eqlm, house_length_lu, house_length_pu, house_width_pu, roof_angle, \
    eg_height_pu, roof_height_pu, overhang_pu, domain_length_pu, domain_width_pu, domain_height_pu, inlet_bc, outlet_bc, \
    ground_bc, house_bc, top_bc, combine_solids = \
    [args[_] for _ in ["name", "default_device", "float_dtype", "t_sim_max", "cluster", "outdir", "outdir_vtk",
                       "vtk", "vtk_fps", "vtk_interval", "vtk_start", "nan_reporter", "from_cpt", "sim_i",
                       "write_cpt", "re", "ma", "viscosity_pu", "char_density_pu", "u_init", "n_steps", "t_target",
                       "step_start", "collision", "dim", "stencil", "eqlm", "house_length_lu", "house_length_pu",
                       "house_width_pu", "roof_angle", "eg_height_pu", "roof_height_pu", "overhang_pu", "domain_length_pu",
                       "domain_width_pu", "domain_height_pu", "inlet_bc", "outlet_bc", "ground_bc", "house_bc",
                       "top_bc", "combine_solids"]]

# CREATE timestamp, sim-ID, outdir and outdir_vtk
timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
sim_id = str(timestamp) + "-" + name
os.makedirs(outdir+"/"+sim_id)
if outdir_vtk is None:
    outdir_vtk = outdir
outdir_vtk = outdir_vtk+"/"+sim_id
if vtk and not os.path.exists(outdir_vtk):
    os.makedirs(outdir_vtk)
print(f"Outdir/simID = {outdir}/{sim_id}")
outdir = outdir+"/"+sim_id  # adding individal sim-ID to outdir path to get individual DIR per simulation
print(f"Input arguments: {args}")

# START LOGGER -> get all terminal output into file
old_stdout = sys.stdout
sys.stdout = Logger(outdir)

### HAUS Flow
# Domain XYZ LU (ODER RES und Dmain in PU (!) -> Philipp hat das so, daraus kann man shape auch bestimmen)
# RE, MA, lattice, char_length_lu, char_length_pu, char_velocity_pu
# MK: char_density_pu (!)?
# u_init (Geschw.feld der Init)
# ref_height? -> obs eg oder roof ist...
# BCs: top (ZG, EQ), bottom (FW,HW,IBB), house(FW,HW,IBB,full (one solid)), inlet (EQ, (NEX), SEI), outlet (EQoP, EXO?), (sides=periodic, ZG)
# SEI: K_Factor, L, N
#
# ? übergebe ich das WSP als Methode, als Werte oder als "string"?

#pspelt ?
# shape ist LU
# domain_constraints sind PU
# grid sind die PU-Koordinaten der Gridpunte -> das ist bei mir auch so! :)
# res ist der LU/PU Umrechungsfaktor
# depth?, debug?, fwbb, house?, parallel?, cluster?
# initial_solution returnt tensoren p und u

## Boundary Masken/Positionen und Trennung/Sortierung
# - inlet, outlet, top, bottom/ground, house
# (!) house-mask jetzt über Philipps kram bzw. resultiert die collision_data!
# (?) möchte ich die FWBB, HWBB auch auf "collision_data" anpassen?
# (!) behandle Ecken/Kanten sorgfältig, insb. auch bzgl. der nicht-lokalen Boundaries
# (!) behandle zwei "nahe", berührende oder überlagernde Solid-Boundaries!
#   - IDEE: solid_maske und solid_surface Maske? -> muss man das definieren,
#   ...oder kann man die solid_masken der Boundaries mit den f_index Listen vergleichen und schauen,
#   ...ob die f_index[i] in einem anderen solid liegt? Dann kann man die entsprechend aus der Liste für bounce und force rausnehmen

# "house" Methode und Maske sollen mit Philipps Code erstellt werden, die MK-Haus-Methoden können raus

# (!) add_boundary nötig?
#   - add_boundary wird im "Main" aufgerufen, nicht in boundary... - brauche ich es? welcher Vorteil? -

## Solid Boundary Conditions
# - solid_maske (wo ist solid)
#   - Vereinigung aller solid_masks der Boundaries? - jeweils on demand? Attribut?
# - IBB: ds... mit f_index_liste
# - surface-Maske? oder bounce_maske, force-maske
#   - bounce - relevant könten sein: Pops. die gelesen werden, Pops. die geschrieben werden
#   - force - relevant alle "Knoten" bzw. genauer Pops. die gelesen werden

## WSP
# - brauche Nullhöhe
# - brauche Referenzhöhe und Referenzgeschwindigkeit
# (?) wird "ab der Nullhöhe", bzw. "ab dem ersten Knoten mit ux>0" zurückgegeben, oder immer das komplettte Profil, inkl. der Nullen unten?
# INPUT: z(Höhenwerte-Feld) z0, z_ref, u_ref, alpha
# OUTPUT: u-Feld in abhängigkeit von z(Höhenwertefeld) in gleichem Format und shape

## PHILIPP ür index-cleanup: f_index = f_index[torch.where(~solid_mask[f_index[:, 1],f_index[:, 2],f_index[:, 3] if d==3 else None])]

### ANALYSE and PROCESS PARAMETERS

# calc. characteristic velocity from re, viscosity and char_length
char_velocity_pu = re * viscosity_pu / house_length_pu

# calculate resolution LU/m
res = house_length_lu/house_length_pu

# TODO: make n_steps / target, reached and t_start / target / reached consistent!
# - args, NAME als variable, Berechnung, checkpointing, Nutzung durch sim.step(...)

n_start = step_start
n_stop_target = step_start+n_steps
t_start = 0
t_stop_target = t_target
# infer step number from target PU-time
if t_target > 0:  # if t_target (PU) is given, calculate n_stop_target from LU/PU length and velocity coefficients
    n_stop_target = int(t_stop_target * house_length_lu/house_length_pu * char_velocity_pu/(ma*1/np.sqrt(3)))
    n_steps = n_stop_target - n_start
else:
    t_target = n_stop_target / (house_length_lu/house_length_pu * char_velocity_pu/(ma*1/np.sqrt(3)))
    t_stop_target = t_target
print(f"(INFO) trying to simulate {n_steps} ({n_start} to {n_stop_target}) steps, representing {t_stop_target:.3f} seconds [PU]!")

### SIMULATOR SETUP
print("STATUS: simulator setup started")
# ceate objects, link and assemble

# STENCIL
if dim == 2:
    if stencil == "D2Q9":
        stencil_obj = lt.D2Q9
    else:
        print("WARNING: wrong stencil choice for 2D simulation, D2Q9 is used")
        stencil_obj = lt.D2Q9
elif dim == 3:
    if stencil == "D3Q15":
        stencil_obj = lt.D3Q15
    elif stencil == "D3Q19":
        stencil_obj = lt.D3Q19
    elif stencil == "D3Q27":
        stencil_obj = lt.D3Q27
    else:
        print("WARNING: wrong stencil choice for 3D simulation, D3Q27 is used")
        stencil_obj = lt.D3Q27
else:
    print("WARNING: wrong dimension choise. Using 2D simulation and D2Q9")
    stencil_obj = lt.D2Q9
    dim = 2

if float_dtype == "foat32" or float_dtype == "single":
    float_dtype = torch.float32
elif float_dtype == "double" or float_dtype == "float64":
    float_dtype = torch.float64
elif float_dtype == "half" or float_dtype == "float16":
    float_dtype = torch.float16

# LATTICE
lattice = lt.Lattice(stencil_obj, device=torch.device(default_device), dtype=float_dtype)
if eqlm:  # use EQLM with 20% less memory usage and 2% less performance
    print("(INFO) using Equilibrium_LessMemory (saving ~20% VRAM on GPU, but ~2% slower)")
    lattice.equilibrium = lt.QuadraticEquilibrium_LessMemory(lattice)

# HOUSE FLOW and DOMAIN
print("Defining house geometry and position")
house_position = (domain_length_pu/3, domain_width_pu/2) if dim == 3 else domain_length_pu/3
if house_width_pu == 0:
    house_width_pu = house_length_pu

roof_length_pu = house_length_pu
if overhang_pu != 0:
    roof_length_pu = house_length_pu + 2*overhang_pu

# infer roof_height and eg_height from another and determine reference height, for u_char and wind speed profile
if roof_height_pu == 0: # if roof_height ist not given infer it from eg_height
    if eg_height_pu == 0: # if nothing is given house is square/cube shaped plus roof
        eg_height_pu = house_length_pu
    roof_height_pu = eg_height_pu + 0.001 + np.tan(roof_angle) * roof_length_pu/2
    reference_height_pu = eg_height_pu
else:
    eg_height_pu = roof_height_pu - (house_length_pu/2 + overhang_pu) * np.tan(roof_angle)
    reference_height_pu = roof_height_pu




# DOMAIN constraints in PU
print("defining domain constraints")
xmin, ymin, zmin = 0, 0, 0 if dim == 3 else None
xmax, ymax, zmax = domain_length_pu, domain_height_pu, domain_width_pu if dim == 3 else None
minz_house, maxz_house = (domain_width_pu/2.-house_width_pu/2., domain_width_pu/2.+house_width_pu/2.) if dim == 3 else (-1, 1)
ground_height_pu = 0.5 * house_length_pu/house_length_lu  # height of ZERO-height or ground level in PU at 0.5 LU

domain_constraints = ([xmin, ymin], [xmax, ymax]) if dim == 2 else ([xmin, ymin, zmin], [xmax, ymax, zmax])  # Koordinatensystem abh. von der stl und deren ursprung
lx, ly, lz = xmax-xmin, ymax-ymin, zmax-zmin  # das sind die PU-Domänengrößen
shape = (int(round(lx*res)), int(round(ly*res))) if dim == 2 else (int(round(lx*res)), int(round(ly*res)), int(round(lz*res))) # LU-Domänengröße

print(f"domain constraints = {domain_constraints}")
print(f"shape = {shape}")

# house_polygon = [[15, 0+ground_height_pu], [15, 10+ground_height_pu], [14, 10+ground_height_pu], [20, 15.5+ground_height_pu],
#                      [26, 10+ground_height_pu], [25, 10+ground_height_pu], [25, 0+ground_height_pu]]
print("defining house and ground 2D polygon")
house_polygon = [[house_position[0]-house_length_pu/2, ground_height_pu*0.999],  # bottom left (slightly lowered into ground for easy combination)
                 [house_position[0]-house_length_pu/2, eg_height_pu+ground_height_pu],  # top left eg
                 [house_position[0]-house_length_pu/2-overhang_pu, eg_height_pu+ground_height_pu],  # top left roof
                 [house_position[0], roof_height_pu+ground_height_pu],  # center roof top
                 [house_position[0]+house_length_pu/2+overhang_pu, eg_height_pu+ground_height_pu],  # top right roof
                 [house_position[0]+house_length_pu/2, eg_height_pu+ground_height_pu],  # top right eg
                 [house_position[0]+house_length_pu/2, ground_height_pu*0.999]]  # bottom right (slightly lowered into ground for easy combination)
house_polygon = [[15, 0.16666666666666666], [15, 10.166666666666666], [14, 10.166666666666666], [20, 15.666666666666666], [26, 10.166666666666666], [25, 10.166666666666666], [25, 0.16666666666666666]]
# TODO: make polygon variable again
# ground_polygon (rectangle, with corners outside the domain, to ease neighbor-search over domain borders
ground_polygon = [[xmin-0.1*domain_length_pu, ground_height_pu],  # top left
                  [xmax+0.1*domain_length_pu, ground_height_pu],  # top right
                  [xmax+0.1*domain_length_pu, ymin-0.1*domain_length_pu],  # bottom right
                  [xmin-0.1*domain_length_pu, ymin-0.1*domain_length_pu]]  # bottom left
                # alles außer der Höhe "außerhalb" der boundary setzen ("ich setze der Nachbarsuche, dort etwas hin, was er findet")


## CALCULATE COLLISION DATA
print("calculating 3D TopoDS_Shapes...")
house_bc_name = "house_BC_name_palceholder"
ground_bc_name = "ground_BC_name_palceholder"

house_prism_shape = build_house_max(house_polygon, minz=minz_house, maxz=maxz_house)  #TopoDS_Shape als Rückgabe
if combine_solids:
    print("(INFO) combining Shapes of house and ground")
    ground_prism_shape = build_house_max(ground_polygon, minz=zmin-0.1*domain_width_pu, maxz=zmax+0.1*domain_width_pu)
    house_prism_shape = TopoDS_Shape(BRepAlgoAPI_Fuse(house_prism_shape, ground_prism_shape).Shape())

# (opt.) combine house and ground solid objects to single object

# TODO: passe no_store_coll und collision_data_path und recalc und parallel an...
print("calculating house_solid_boundary_data...")
house_solid_boundary_data = getIBBdata(house_prism_shape, torch.tensor(np.array(makeGrid(domain_constraints, shape)), device=lattice.device), # TODO: clean the tensor(array() stuff with stack/cat etc.
                                        lattice, no_store_coll=False, res=res, dim=dim, name=house_bc_name,
                                        coll_data_path=args["collision_data_path"], redo_calculations=args["recalc"], # TODO: redo_calc as parameter of house3D script
                                        parallel=False, # TODO: eliminate parallelism
                                        device=default_device,
                                        cluster=cluster  #TODO: eliminate cluster tag?
                                        )
ground_solid_boundary_data = None
if not combine_solids:
    print("(INFO) calculating ground_solid_boundary_data...")
    ground_solid_boundary_data = getIBBdata(ground_prism_shape, torch.tensor(np.array(makeGrid(domain_constraints, shape)), device=lattice.device), # TODO: clean the tensor(array() stuff with stack/cat etc.
                                        lattice, no_store_coll=False, res=res, dim=dim, name=ground_bc_name,
                                        coll_data_path=args["collision_data_path"], redo_calculations=True, # TODO: redo_calc as parameter of house3D script
                                        parallel=False, # TODO: eliminate parallelism
                                        device=default_device,
                                        cluster=cluster  #TODO: eliminate cluster tag?
                                        )

# inspect solid_boundary/IBB-data
if args["plot_intersection_info"]:
    print("(INFO) plotting intersection info...")
    plot_intersection_info(house_solid_boundary_data, torch.tensor(makeGrid(domain_constraints, shape), device=lattice.device), lattice, house_solid_boundary_data.solid_mask, outdir, name=house_bc_name)
    if not combine_solids:
        plot_intersection_info(house_solid_boundary_data, torch.tensor(makeGrid(domain_constraints, shape), device=lattice.device), lattice, house_solid_boundary_data.solid_mask, outdir, name=ground_bc_name)

## FLOW Class
print("initializing flow class...")
flow = HouseFlow3D(shape, re, ma, lattice, domain_constraints,
                   char_length_lu=house_length_lu,
                   char_length_pu=house_length_pu,
                   char_velocity_pu=char_velocity_pu,
                   u_init=0,
                   reference_height_pu=reference_height_pu, ground_height_pu=ground_height_pu,
                   inlet_bc=inlet_bc, outlet_bc=outlet_bc,
                   ground_bc=ground_bc if not combine_solids else None,
                   house_bc=house_bc, top_bc=top_bc,
                   house_solid_boundary_data=house_solid_boundary_data,
                   ground_solid_boundary_data=ground_solid_boundary_data,
                   K_Factor=10,  # K_factor for SEI boundary inlet
                   L=3,  # L for SEI
                   N=34,  # N number of random voctices for SEI
                   )


# COLLISION
print("initializing collision operator...")
collision_obj = None
if collision.casefold() == "reg" or collision.casefold() == "bgk_reg":
    collision_obj = lt.RegularizedCollision(lattice, tau=flow.units.relaxation_parameter_lu)
elif collision.casefold() == "kbc":
    if dim == 2:
        collision_obj = lt.KBCCollision2D(lattice, tau=flow.units.relaxation_parameter_lu)
    else:
        collision_obj = lt.KBCCollision3D(lattice, tau=flow.units.relaxation_parameter_lu)
else:  # default to bgk
    collision_obj = lt.BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)

print("initializing streaming object...")
streaming = lt.StandardStreaming(lattice)
print("initializing simulation object...")
simulation = lt.Simulation(flow, lattice, collision_obj, streaming)

# OTPIONAL
#simulation.initialize_f_neq()
grid_reynolds_number = flow.units.characteristic_velocity_lu/(lattice.stencil.cs**2 * (flow.units.relaxation_parameter_lu - 0.5))  # RE_grid as a measure for free flow resolution (should be ~O(10) )
print(f"grid Reynolds number Re_grid = {grid_reynolds_number}")

## CHECK INITIALISATION AND 2D DOMAIN
#TODO: checkout initialisation by printing a 2D slice through the domain and display ux
#TODO: print 2D mask of solid for 2D slice through domain


## REPORTER
# create and append reporters
print("initializing reporters")

# OBSERVABLE REPORTERS
max_u_lu_observable = lt.MaximumVelocityLU(lattice,flow)
max_u_lu_reporter = lt.ObservableReporter(max_u_lu_observable, interval=1, out=None)
simulation.reporters.append(max_u_lu_reporter)

# VTK REPORTER
if vtk:
    print(f"(INFO) Appending vtk reporter wit vtk_interval = {int(flow.units.convert_time_to_lu(1/vtk_fps)) if vtk_interval == 0 else int(vtk_interval)} and vtk_dir: {outdir_vtk}/vtk/out")
    vtk_reporter = lt.VTKReporter(lattice, flow,
                                  interval=int(flow.units.convert_time_to_lu(1/vtk_fps)) if vtk_interval == 0 else int(vtk_interval),
                                  filename_base=outdir_vtk+"/vtk/out")
    simulation.reporters.append(vtk_reporter)
# TODO: obstacle point und cell Maske ausgeben für VTK output
# MK hat da auch den "ouput_mask" zum vtk_reporter hinzugefügt und kann das nach der Initialisierung aufrufen

# NAN REPORTER
nan_reporter = lt.NaNReporter(flow,lattice,n_stop_target, t_stop_target)
simulation.reporters.append(nan_reporter)
# TODO: flag in simulation, welcher den Step regulär beendet, wenn a) Zeit erreicht, oder b) NaN detektiert wurde.
#       ... Idee: simulation besitzt einen flag "abort" oder "stop" und ein "reason" (string), in den die Reporter schreiben können.
#       ...Dafür werden die Variablen dem Reporter als simulation.variable übergeben, dann muss man nicht komplett simulation übergeben.

## WRITE PARAMETERS to file in outdir
# TODO: write parameters to file (ähnlich zu mdout.mdp)


## LOAD CHECKPOINT FILE
# TODO: load checkpoint file and adjust sim.i


### RUN SIMULATION
#n_steps = n_stop_target - n_start
t_start = time()

print("***** RUNNING SIMULATION *****")
mlups = simulation.step(n_steps)

t_end = time()
runtime = t_end-t_start

# PRINT SOME STATS TO STDOUT:
print("MLUPS:", mlups)
print("PU-Time: ",flow.units.convert_time_to_pu(n_steps)," seconds")
print("number of steps:", n_steps)
print("runtime: ", runtime, "seconds (", round(runtime/60, 2), "minutes )")

print("current GPU VRAM (MB): ", torch.cuda.memory_allocated(device=default_device)/1024/1024)
print("max. GPU VRAM (MB): ", torch.cuda.max_memory_allocated(device=default_device)/1024/1024)

[cpuLoad1,cpuLoad5,cpuLoad15] = [x / psutil.cpu_count() * 100 for x in psutil.getloadavg()]
print("CPU % avg. over last 1 min, 5 min, 15 min; ", round(cpuLoad1,2), round(cpuLoad5,2), round(cpuLoad15,2))

ram = psutil.virtual_memory()
print("current total RAM usage [MB]: " + str(round(ram.used/(1024*1024),2)) + " of " + str(round(ram.total/(1024*1024),2)) + " MB")

u = lattice.u(simulation.f)
print(f"u.shape post: {u.shape}")
rho = lattice.rho(simulation.f)
print(f"rho.shape post {rho.shape}")
u = flow.units.convert_velocity_to_pu(u).cpu().detach().numpy()[:,:,:,int(u.shape[3]/2)]
p = flow.units.convert_density_lu_to_pressure_pu(rho).cpu().detach().numpy()[:,:,:,int(rho.shape[3]/2)]
unorm = np.linalg.norm(u, axis=0)

# Plot without outliers due to bounce-back contacts
fig, axes = plt.subplots(1,2, figsize=(12,6), dpi=300)
fig.tight_layout()
axes[0].set_title("Pressure")
axes[0].imshow(p[0].transpose(), origin="lower", )#vmin=p[0,ny-1,0], vmax=p[0].mean(axis=-1).max())
axes[1].set_title("Velocity magnitude")
axes[1].imshow(unorm.transpose(), origin="lower", cmap='inferno',
               vmin=np.percentile(unorm.flatten(),1),
               vmax=np.percentile(unorm.flatten(),95)
              )
plt.show()

### EXPORT STATS
# TODO: export stats to file
# TODO: OUTUPT CUDA-VRAM-DATA

### WRITE CHECKPOINT
# TODO: write checkpointfile, if write_cpt=True and save sim.i to it / should be on correct device! or CPU to be save

### POSTPROCESSONG: PROCESS and export observables
# TODO: Obsrevable post-processing, while flow and sim data is still available

### OUTPUT RESULTS
# TODO: output results to human-readable file AND to copy-able file (or csv)

### SAVE SCRIPT: save this script to outdir
# TODO: save this script to outdir for later reproducibility and reference

sys.stdout = old_stdout