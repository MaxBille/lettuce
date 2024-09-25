### IMPORT
import os
import hashlib
import resource
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from time import time, sleep
from math import ceil, floor, sqrt
import datetime

import numpy as np
import psutil
import torch
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse

import lettuce as lt

from pspelt.geometric_building_model import build_house_max
from pspelt.helperFunctions.getIBBdata import getIBBdata
from pspelt.helperFunctions.plotting import plot_intersection_info
from pspelt.helperFunctions.logging import Logger
from pspelt.obstacleFunctions import makeGrid

### ARGUMENTS

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument("--name", default="3Dhouse", help="name of the simulation, appears in output directory name")
parser.add_argument("--float_dtype", default="float32", choices=["float32", "float64", "single", "double", "half"], help="data type for floating point calculations in torch")
parser.add_argument("--cluster", action='store_true', help="if you don't want pngs etc. to open, please use this clsuter-flag")
parser.add_argument("--dim", default=3, type=int, help="dimensions: 2D (2), oder 3D (3, default)")
parser.add_argument("--stencil", default="D3Q27", choices=['D2Q9', 'D3Q15', 'D3Q19', 'D3Q27'], help="stencil (D2Q9, D3Q27, D3Q19, D3Q15), dimensions will be infered from D")

parser.add_argument("--outdir", default=os.getcwd(), type=str, help="directory to save output files to; vtk-files will be saved in seperate dir, if outputdir_vtk is specified")

# house and domain geometry
parser.add_argument("--house_length_lu", default=10, type=int, help="house length in LU")  # characteristic length LU, in flow direction
parser.add_argument("--ground_height_lu", default=0.5, type=float, help="ground height in LU, height ZERO, in absolute coordinates relative to coordinate system")
parser.add_argument("--house_length_pu", default=10, type=float, help="house length in PU")  # characteristic length PU [m]
parser.add_argument("--house_width_pu", default=0, type=float, help="width of house in crossstream direction. If left default, it will be equal to house_length_pu")  # cross-stream house_width PU [m]
#house_position  # center of house foundation (corner closest to domain origin?) / erstmal hardcoded, denn man kann als argument wohl kein tupel übergeben
parser.add_argument("--roof_angle", default=45, type=float, help="roof_angle in degree (0 to <90)")  # angle of roof (incline and decline for symmetric roof) - depending on how the house-polygon is defined, obsolete?
parser.add_argument("--eg_height_pu", default=0, type=float, help="eg_height in PU")  # if left 0, roof_height is taken. If roof_height is zero as well, eg_height_pu = house_length_pu
parser.add_argument("--roof_height_pu", default=0, type=float, help="roof_height in PU") # if left 0, roof_height is infered from eg_height and roof_angle
parser.add_argument("--overhang_pu", default=0, type=float, help="roof overhang in PU")
parser.add_argument("--domain_length_pu", default=60, type=float, help="flow-direction domain length in PU")
parser.add_argument("--domain_width_pu", default=40, type=float, help="cross-flow-direction domain width in PU")
parser.add_argument("--domain_height_pu", default=30, type=float, help="cross-flow domain height in PU")

parser.add_argument("--combine_solids", action='store_true', help="combine all solids (house and ground) into one object for easier prototyping")
parser.add_argument("--no_house", action='store_true', help="if TRUE, removes house from simulation, for debugging of house-independent aspects")
parser.add_argument("--solid_boundary_data_path", default=os.path.join(os.getcwd(), 'solid_boundary_data'), type=str, help="")  # DAS BRAUCH ICH...
parser.add_argument("--no_store_solid_boundary_data", action='store_true', help="") # ob coll_data gespeichert wird, oder nicht... -> ohne, wirds zwar verwendet, aber nicht gespeichert
parser.add_argument("--recalc", action='store_true', help="recalculate solid_boundary_data") # DAS BRAUCHE ICH AUCH
parser.add_argument("--plot_intersection_info", action='store_true', help="plot intersection info to outdir to debug solid-boundary problems")
parser.add_argument("--verbose", action='store_true', help="display more information in console (for example about neighbour search)")

args = vars(parser.parse_args())

time0 = time()

default_device = "cpu"

# get parameters from args[] dict:
name, float_dtype, cluster, outdir, dim, stencil, house_length_lu, house_length_pu, house_width_pu, roof_angle, \
    eg_height_pu, roof_height_pu, overhang_pu, domain_length_pu, domain_width_pu, domain_height_pu, combine_solids, verbose = \
    [args[_] for _ in ["name", "float_dtype", "cluster", "outdir", "dim", "stencil", "house_length_lu", "house_length_pu",
                       "house_width_pu", "roof_angle", "eg_height_pu", "roof_height_pu", "overhang_pu", "domain_length_pu",
                       "domain_width_pu", "domain_height_pu", "combine_solids", "verbose"]]

# CREATE timestamp, sim-ID, outdir and outdir_vtk
timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
dir_id = str(timestamp) + "-" + name
os.makedirs(outdir+"/"+dir_id)
print(f"Outdir/simID = {outdir}/{dir_id}")
outdir = outdir+"/"+dir_id # adding individal sim-ID to outdir path to get individual DIR per simulation
print(f"Input arguments: {args}")

# START LOGGER -> get all terminal output into file
old_stdout = sys.stdout
sys.stdout = Logger(outdir)

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

if float_dtype == "float32" or float_dtype == "single":
    float_dtype = torch.float32
elif float_dtype == "double" or float_dtype == "float64":
    float_dtype = torch.float64
elif float_dtype == "half" or float_dtype == "float16":
    float_dtype = torch.float16

# LATTICE
lattice = lt.Lattice(stencil_obj, device=torch.device(default_device), dtype=float_dtype)


# calculate resolution LU/m
res = house_length_lu/house_length_pu

# HOUSE FLOW and DOMAIN
print("Defining house geometry and position...")
house_position = (domain_length_pu/3, domain_width_pu/2) if dim == 3 else domain_length_pu/3
if house_width_pu == 0:
    print(f"(INFO) house_width_pu == 0: implies house_width = house_length!")
    house_width_pu = house_length_pu

roof_length_pu = house_length_pu
if overhang_pu != 0:
    roof_length_pu = house_length_pu + 2*overhang_pu

# infer roof_height and eg_height from another and determine reference height, for u_char and wind speed profile
if roof_height_pu == 0: # if roof_height ist not given infer it from eg_height
    if eg_height_pu == 0: # if nothing is given house is square/cube shaped plus roof
        eg_height_pu = house_length_pu
        print(f"(INFO) roof_height == eg_height == 0: implies eg_height = house_length (char. length) -> cubic EG-XY-shape")
    roof_height_pu = eg_height_pu * (1 + 0) + np.tan(roof_angle*np.pi/180) * roof_length_pu/2 #0.01*house_length_pu/house_length_lu
    print(f"(INFO) roof_height == 0: calculating roof_height from eg_height: eg_height_pu = {eg_height_pu}, roof_angle = {roof_angle}, roof_length_pu = {roof_length_pu}")
    reference_height_pu = eg_height_pu
    print(f"(INFO) Setting reference_height = eg_height")
else:
    eg_height_pu = roof_height_pu - np.tan(roof_angle*np.pi/180) * roof_length_pu/2
    print(f"(INFO) eg_height == 0: calculating eg_height from roof_height: roof_height_pu = {roof_height_pu}, roof_angle = {roof_angle}")
    reference_height_pu = roof_height_pu
    print(f"(INFO) Setting reference_height = roof_height")


# DOMAIN constraints in PU
print("Defining domain constraints...")
xmin, ymin, zmin = 0, 0, 0 if dim == 3 else None
xmax, ymax, zmax = domain_length_pu, domain_height_pu, domain_width_pu if dim == 3 else None
minz_house, maxz_house = (domain_width_pu/2.-house_width_pu/2., domain_width_pu/2.+house_width_pu/2.) if dim == 3 else (-1, 1)
ground_height_pu = args["ground_height_lu"] * house_length_pu/house_length_lu  # height of ZERO-height or ground level in PU at 0.5 LU

domain_constraints = ([xmin, ymin], [xmax, ymax]) if dim == 2 else ([xmin, ymin, zmin], [xmax, ymax, zmax])  # Koordinatensystem abh. von der stl und deren ursprung
lx, ly, lz = xmax-xmin, ymax-ymin, zmax-zmin  # das sind die PU-Domänengrößen
shape = (int(round(lx*res)), int(round(ly*res))) if dim == 2 else (int(round(lx*res)), int(round(ly*res)), int(round(lz*res))) # LU-Domänengröße

print(f"-> Domain PU constraints = {domain_constraints}")
print(f"-> Domain LU shape = {shape}")

#TODO: passe house_position (PU!) in Abh. der Länge und Breite der Domain so an, dass das Zentrum MITTIG ZWISCHEN, oder genau AUF Knoten liegt => SYMMETRIE (!)
#  u.u. muss dafür anders aus domain_length etc. gerechnet werden. -> siehe nochmal Zylinder

# house_polygon = [[15, 0+ground_height_pu], [15, 10+ground_height_pu], [14, 10+ground_height_pu], [20, 15.5+ground_height_pu],
#                      [26, 10+ground_height_pu], [25, 10+ground_height_pu], [25, 0+ground_height_pu]]
print("Defining house and ground 2D polygon for OCC solid generation...")
house_polygon = [[house_position[0]-house_length_pu/2, ground_height_pu*0.999],  # bottom left (slightly lowered into ground for easy combination)
                 [house_position[0]-house_length_pu/2, eg_height_pu+ground_height_pu],  # top left eg
                 [house_position[0]-house_length_pu/2-overhang_pu, eg_height_pu+ground_height_pu],  # top left roof
                 [house_position[0], roof_height_pu+ground_height_pu],  # center rooftop
                 [house_position[0]+house_length_pu/2+overhang_pu, eg_height_pu+ground_height_pu],  # top right roof
                 [house_position[0]+house_length_pu/2, eg_height_pu+ground_height_pu],  # top right eg
                 [house_position[0]+house_length_pu/2, ground_height_pu*0.999]]  # bottom right (slightly lowered into ground for easy combination)
#house_polygon = [[15, 0.16666666666666666], [15, 10.166666666666666], [14, 10.166666666666666], [20, 15.666666666666666], [26, 10.166666666666666], [25, 10.166666666666666], [25, 0.16666666666666666]]  # TEST with consts
# ground_polygon (rectangle, with corners outside the domain, to ease neighbor-search over domain borders
ground_polygon = [[xmin-0.1*domain_length_pu, ground_height_pu],  # top left
                  [xmax+0.1*domain_length_pu, ground_height_pu],  # top right
                  [xmax+0.1*domain_length_pu, ymin-0.1*domain_length_pu],  # bottom right
                  [xmin-0.1*domain_length_pu, ymin-0.1*domain_length_pu]]  # bottom left
                # alles außer der Höhe "außerhalb" der boundary setzen ("ich setze der Nachbarsuche, dort etwas hin, was er findet")


# create unique ID of geometry parameters:
geometry_hash = hashlib.md5(f"{combine_solids}{args['no_house']}{domain_constraints}{shape}{house_position}{ground_height_pu}{house_length_pu}{house_length_pu}{eg_height_pu}{house_width_pu}{roof_height_pu}{overhang_pu}".encode()).hexdigest()
house_bc_name = "house_BC_"+ str(geometry_hash)
ground_bc_name = "ground_BC_" + str(geometry_hash)

# SAVE geometry input to file:
output_file = open(outdir+"/geometry_pu.txt", "a")
output_file.write(f"\nGEOMETRY of house and ground, after inference of missing lengths (see log):\n")
output_file.write(f"\ncombine_solids = {combine_solids}")
output_file.write(f"\nno_house = {args['no_house']}")
output_file.write(f"\ndx_pu [m] = {(house_length_pu/house_length_lu):.4f}")
output_file.write(f"\ndomain_constraints PU = {domain_constraints}")
output_file.write(f"\ndomain shape LU = {shape}")
output_file.write(f"\nhouse_position_PU (on XZ ground plane) = {house_position}")
output_file.write(f"\nground_height PU = {ground_height_pu:.4f}")
output_file.write(f"\nhouse_length LU = {house_length_lu}")
output_file.write(f"\nhouse_length PU = {house_length_pu:.4f}")
output_file.write(f"\nhouse width PU = {house_width_pu:.4f}")
output_file.write(f"\neg height PU = {eg_height_pu:.4f}")
output_file.write(f"\nroof height PU = {roof_height_pu:.4f}")
output_file.write(f"\nroof angle = {roof_angle:.4f}")
output_file.write(f"\noverhangs PU = {overhang_pu:.4f}")
output_file.write(f"\n\ngeometry hash: {geometry_hash}")
output_file.close()

## CALCULATE SOLID BOUNDARY DATA
print("Calculating 3D TopoDS_Shapes...")

house_prism_shape = build_house_max(house_polygon, minz=minz_house, maxz=maxz_house)  #TopoDS_Shape als Rückgabe
ground_prism_shape = build_house_max(ground_polygon, minz=zmin-0.1*domain_width_pu, maxz=zmax+0.1*domain_width_pu)

time1 = time()
print(f"(TIME) Calculating TopoDS_Shapes took {floor((time1-time0) / 60):02d}:{floor((time1-time0) % 60):02d} [mm:ss]")

if combine_solids:
    print("(INFO) combine_solids==True -> Combining Shapes of house and ground...")
    house_prism_shape = TopoDS_Shape(BRepAlgoAPI_Fuse(house_prism_shape, ground_prism_shape).Shape())
    time11 = time()
    print(
        f"(TIME) Combining TopoDS_Shapes took {floor((time11 - time1) / 60):02d}:{floor((time11 - time1) % 60):02d} [mm:ss]")

time1 = time()

# (opt.) combine house and ground solid objects to single object

# TODO: passe no_store_coll und solid_boundary_data_path und recalc und parallel an...
house_solid_boundary_data = None
if not args["no_house"]:
    print("(INFO) Calculating house_solid_boundary_data...")
    house_solid_boundary_data = getIBBdata(house_prism_shape, makeGrid(domain_constraints, shape), periodicity=(False, False, True), # TODO: clean the tensor(array() stuff with stack/cat etc.
                                           lattice=lattice, no_store_solid_boundary_data=False, res=res, dim=dim, name=house_bc_name,
                                           solid_boundary_data_path=args["solid_boundary_data_path"], redo_calculations=args["recalc"],  # TODO: redo_calc as parameter of house3D script
                                           parallel=False,  # TODO: eliminate parallelism
                                           device=default_device,
                                           cluster=cluster,
                                           verbose=verbose
                                           )
ground_solid_boundary_data = None
if not combine_solids:
    print("(INFO) Calculating ground_solid_boundary_data...")
    ground_solid_boundary_data = getIBBdata(ground_prism_shape, makeGrid(domain_constraints, shape),  periodicity=(False, False, True), # TODO: clean the tensor(array() stuff with stack/cat etc.
                                            lattice=lattice, no_store_solid_boundary_data=False, res=res, dim=dim, name=ground_bc_name,
                                            solid_boundary_data_path=args["solid_boundary_data_path"], redo_calculations=args["recalc"],  # TODO: redo_calc as parameter of house3D script
                                            parallel=False,  # TODO: eliminate parallelism
                                            device=default_device,
                                            cluster=cluster,
                                            verbose=verbose
                                            )

time2 = time()

# inspect solid_boundary/IBB-data
if args["plot_intersection_info"]:
    if not args["no_house"]:
        print("(INFO) plotting intersection info...")
        plot_intersection_info(house_solid_boundary_data, makeGrid(domain_constraints, shape), lattice, house_solid_boundary_data.solid_mask, outdir, name=house_bc_name, show=not cluster)
        if not combine_solids:
            plot_intersection_info(ground_solid_boundary_data, makeGrid(domain_constraints, shape), lattice, ground_solid_boundary_data.solid_mask, outdir, name=ground_bc_name, show=not cluster)

time_end = time()


print(f"(INFO) Total runtime of SBD creation: {floor((time_end - time0) / 60):02d}:{floor((time_end - time0) % 60):02d} [mm:ss]")
print("maximum total (CPU) RAM usage ('MaxRSS') [MB]: " + str(round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, 2)) + " MB")
[cpuLoad1,cpuLoad5,cpuLoad15] = [x / psutil.cpu_count() * 100 for x in psutil.getloadavg()]
print("CPU LOAD AVG.-% over last 1 min, 5 min, 15 min; ", round(cpuLoad1, 2), round(cpuLoad5, 2), round(cpuLoad15, 2))

## END OF SCRIPT
print(f"\n♬ THE END ♬")
sys.stdout = old_stdout