### IMPORT

# sys, os etc.
import sys
import os
from math import floor

import psutil
import shutil
import hashlib
import resource
from time import time, sleep
import datetime
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from PIL import Image
import subprocess

import numpy as np
import torch
from OCC.Core.TopoDS import TopoDS_Shape
from matplotlib import pyplot as plt
from pyevtk.hl import imageToVTK

# lettuce
import lettuce as lt
from lettuce import torch_gradient
from lettuce.boundary import InterpolatedBounceBackBoundary_occ, BounceBackBoundary
from lettuce.boundary_mk import EquilibriumExtrapolationOutlet, NonEquilibriumExtrapolationInletU, ZeroGradientOutlet, SyntheticEddyInlet
# flow
from lettuce.flows.velocityKeilFlow import VelocityKeilFlow

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
parser.add_argument("--name", default="Keil3D", help="name of the simulation, appears in output directory name")
parser.add_argument("--default_device", default="cuda", type=str, help="run on cuda or cpu")
parser.add_argument("--float_dtype", default="float32", choices=["float32", "float64", "single", "double", "half"], help="data type for floating point calculations in torch")
parser.add_argument("--t_sim_max", default=(71*60*60), type=float, help="max. walltime [s] to simulate, default is 71 h for cluster use. sim stops at 0.99*t_max_sim")  # andere max.Zeit? wie lange braucht das "drum rum"? kann cih simulation auch die scho vergangene Zeit übergeben? dann kann ich mit nem größeren Wert rechnen und sim ist variabel darin, wie viel Zeit es noch hat

parser.add_argument("--cluster", action='store_true', help="if you don't want pngs etc. to open, please use this clsuter-flag")
parser.add_argument("--outdir", default=os.getcwd(), type=str, help="directory to save output files to; vtk-files will be saved in seperate dir, if outputdir_vtk is specified")
parser.add_argument("--outdir_data", default=None, type=str, help="")

parser.add_argument("--vtk_3D", action='store_true')
parser.add_argument("--vtk_3D_fps", type=float)
parser.add_argument("--vtk_3D_step_interval", type=float)
parser.add_argument("--vtk_3D_t_interval", type=float)
parser.add_argument("--vtk_3D_step_start", type=int)
parser.add_argument("--vtk_3D_step_end", type=int)
parser.add_argument("--vtk_3D_t_start", type=float)
parser.add_argument("--vtk_3D_t_end", type=float)


parser.add_argument("--vtk_slice2D", action='store_true', help="toggle vtk-output of 2D slice for WHOLE DOMAIN (!) to outdir_data, if set True (1)")
parser.add_argument("--vtk_slice2D_fps", type=float)
parser.add_argument("--vtk_slice2D_step_interval", type=float)
parser.add_argument("--vtk_slice2D_t_interval", type=float)
parser.add_argument("--vtk_slice2D_step_start", type=int)
parser.add_argument("--vtk_slice2D_step_end", type=int)
parser.add_argument("--vtk_slice2D_t_start", type=float)
parser.add_argument("--vtk_slice2D_t_end", type=float)

# SURVEILANCE reporters
parser.add_argument("--nan_reporter", action='store_true', help="stop simulation if NaN is detected in f field")
parser.add_argument("--nan_reporter_interval", default=100, type=int, help="interval in which the NaN reporter checks f for NaN")
parser.add_argument("--high_ma_reporter", action='store_true', help="stop simulation if Ma > 0.3 is detected in u field")
parser.add_argument("--high_ma_reporter_interval", default=100, type=int, help="interval in which the HighMa reporter checks for Ma>0.3")
parser.add_argument("--watchdog", action='store_true', help="report progress, ETA and warn, if Sim is estimated to run longer than t_max (~72 h)")
parser.add_argument("--watchdog_interval", default=0, type=int, help="interval in which the watchdog reporter reports. 0 sets 100 reports per simulation") #TODO: ist das so korrekt?

# flow physics
parser.add_argument("--re", default=None, type=float, help="Reynolds number")
parser.add_argument("--ma", default=0.05, type=float, help="Mach number (should stay < 0.3, and < 0.1 for highest accuracy. low Ma can lead to instability because of round of errors ")
parser.add_argument("--viscosity_pu", default=14.852989758837 * 10**(-6), type=float, help="kinematic fluid viscosity in PU. Default is air at ~14.853e-6 (at 15°C, 1atm)")
parser.add_argument("--char_density_pu", default=1.2250, type=float, help="density, default is air at ~1.2250 at 15°C, 1atm")  # ist das so korrekt? - von Martin Kiemank übernommen
parser.add_argument("--u_init", default=0, type=int, choices=[0, 1, 2], help="0: initial velocity zero, 1: velocity one uniform, 2: velocity profile") # könnte ich noch auf Philipp und mich anpassen...und uniform durch komplett WSP ersetzen
#OLD: parser.add_argument("--u_max_pu", default=0, type=float, help="max. velocity in PU at the tip of the keil. If set, overwrites Reynoldsnumber!")
# -> see domain_geometry below for velocity-profile parameters
# char velocity PU will be calculated from Re, viscosity and char_length!

# solver settings
parser.add_argument("--n_steps_target", default=None, type=int, help="number of steps to simulate, overwritten by t_target, if t_target is >0, end of sim will be step_start+n_steps")
parser.add_argument("--t_target", default=None, type=float, help="time in PU to simulate, t_start will be calculated by PU/LU-conversion of step_start")
#parser.add_argument("--step_start", default=0, type=int, help="stepnumber to start at. Useful if sim. is started from a checkpoint and sim-data should be concatenated later on")
parser.add_argument("--collision", default="bgk", type=str, choices=["kbc", "bgk", "reg", 'reg', "bgk_reg", 'kbc', 'bgk', 'bgk_reg'], help="collision operator (bgk, kbc, reg)")
#OLD: parser.add_argument("--dim", default=3, type=int, help="dimensions: 2D (2), oder 3D (3, default)")
parser.add_argument("--stencil", default="D3Q27", choices=['D2Q9', 'D3Q15', 'D3Q19', 'D3Q27'], help="stencil (D2Q9, D3Q27, D3Q19, D3Q15), dimensions will be infered from D")
parser.add_argument("--eqlm", action="store_true", help="use Equilibium LessMemory to save ~20% on GPU VRAM, sacrificing ~2% performance")

# domain geometry
#TODO: [opt., Philipp]: wenn die charakteristische Geschwindigkeit der maximalen Geschwindigkeit in der Domäne entspricht, bin ich möglicherweise bzgl. meines Mach-Spielraums begrenzter.
parser.add_argument("--keil_percentage_of_inlet", default=None, type=float, help="percentage of inlet that is keilförmig")
# KEIL STEIGUNG? - vielleicht später, um daraus dann die Reynoldszahl zu berechnen...
parser.add_argument("--keil_delta_ux_pu", default=None, type=float, help="max. delta_ux (grad) for velocity profile in PU. Overwrites Re(!)")
parser.add_argument("--keil_u_max_pu", default=None, type=float, help="max. velocity = characteristic velocity in PU. Overwrites Re and keil_delta_ux_pu (!)")
parser.add_argument("--keil_delta_ux_lu", default=None, type=float, help="max. delta_ux (grad) for velocity profile in LU. Overwrites Re(!)")
parser.add_argument("--keil_u_max_lu", default=None, type=float, help="max. velocity = characteristic velocity in LU. Overwrites Re and keil_delta_ux_lu (!)")
parser.add_argument("--domain_length_x_lu", default=None, type=int, help="domain length in flow direction (X) in LU (only specify lu OR pu)")
parser.add_argument("--domain_length_x_pu", default=None, type=float, help="domain length in flow direction (X) in PU (only specify lu OR pu)")
parser.add_argument("--domain_height_y_lu", default=None, type=int, help="domain height in cross flow direction (Y) in LU (only specify lu OR pu)")
parser.add_argument("--resolution", default=1, type=float, help="number of gridpoints per meter [LU/PU]")
parser.add_argument("--domain_height_y_pu", default=None, type=float, help="domain height in cross flow direction (Y) in PU (only specify lu OR pu)")
parser.add_argument("--domain_width_z_lu", default=None, type=int, help="domain width in cross flow direction (Z) in LU (only specify lu OR pu)")
parser.add_argument("--domain_width_z_pu", default=None, type=float, help="domain width in cross flow direction (Z) in PU (only specify lu OR pu)")

# boundary algorithms
parser.add_argument("--inlet_bc", default="eqin", help="inlet boundary condition: EQin, NEX, SEI")
parser.add_argument("--outlet_bc", default="eqoutp", help="outlet boundary condition: EQoutP, EQoutU")
parser.add_argument("--inlet_ramp_steps", default=1, type=int, help="step number over which the velocity of ramped EquilibriumInlet is ramped to 100%")

# plotting and output
parser.add_argument("--save_animations", action='store_true', help="create and save animations and pngs of u and p fields")
parser.add_argument("--animations_number_of_frames", default=None, type=int, help="number of frames to take over the course of the simulation every t_target/#frames time units, overwrites animations_fps!")
parser.add_argument("--animations_fps_pu", default=None, type=int, help="number of frames per second PU for 2D animations (mp4s). Not the fps for the mp4, but the rate at which frames are taken from simulation (relative to it's simulated PU-time)")
parser.add_argument("--animations_fps_mp4", default=None, type=int, help="number of frames per second PU for 2D animations (mp4s). Actual fps of the resulting mp4. (Not the fps at which frames are taken from simulation!")

args = vars(parser.parse_args())

# CREATE timestamp, sim-ID, outdir and outdir_data
timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
sim_id = str(timestamp) + "-" + args["name"]
os.makedirs(args["outdir"]+"/"+sim_id)
if args["outdir_data"] is None:
    outdir_data = args["outdir"]
else:
    outdir_data = args["outdir_data"]
outdir = args["outdir"]+"/"+sim_id  # adding individal sim-ID to outdir path to get individual DIR per simulation
outdir_data = outdir_data+"/"+sim_id
if (args["vtk_3D"] or args["vtk_slice2D"] or args["save_animations"]) and not os.path.exists(outdir_data):
    os.makedirs(outdir_data)
print(f"Outdir/simID = {outdir}/{sim_id}")
print(f"Outdir_vtk/simID = {outdir_data}/{sim_id}")
print(f"Input arguments: {args}")

# [optional] os.makedirs(outdir+"/PU_point_report")

output_file = open(outdir+"/input_parameters.txt", "a")
for key in args:
    output_file.write('{:30s} {:30s}\n'.format(str(key), str(args[key])))
output_file.close()

if args["cluster"]:
    dpi = 1200
else:
    dpi = 600

### SAVE SCRIPT: save this script to outdir
print(f"\nSaving simulation script to outdir...")
temp_script_name = sim_id + "_" + os.path.basename(__file__)
shutil.copy(__file__, outdir+"/"+temp_script_name)
print(f"Saved simulation script to '{str(outdir+'/'+temp_script_name)}'")

# START LOGGER -> get all terminal output into file
old_stdout = sys.stdout
sys.stdout = Logger(outdir)

def save_mp4(filename: str = "./animation",
             database: str = None,
             dataName: str = None,
             fps: int = 20,
             loop: int = 0):
    """
    Description:
    The save_mp4 function is a helper utility designed to create an animated mp4 from a series of image files.
    The images are read from a specified directory and filtered based on a given chart name. The resulting mp4 is saved
    to a specified output file.

    Parameters:
    - filename (str): The path where the output mp4 will be saved. Default is ./animation.
    - database (str): The directory containing the image files. This should be a valid directory path.
    - dataName (str): A substring to filter the image files in the origin directory. Only files containing this
      substring in their names will be included in the mp4.
    - fps (int): Frames per second for the mp4. This determines the speed of the animation. Default is 20.
    - loop (int): Number of times the mp4 will loop. Default is 0, which means the mp4 will loop indefinitely.
    """
    # Ensure the database path ends with a slash
    database += '/' if database[-1] != "/" else ''

    # List all files in the database directory
    filesInOrigin = sorted(os.listdir(database))

    # Filter files based on the dataName substring
    filesForAnimation = []
    for file in filesInOrigin:
        if dataName in file:
            filesForAnimation.append(file)

    # Print the number of files found
    print(f"(save_mp4): Number of files found: {len(filesForAnimation)}. Creating animation...")

    # TODO (OPT) - Ausgabe der Bilder in eigene Unterordner, sodass dann einfach "alles" aus einem Unterordner von ffmpeg genutzt werden kann...
    if len(filesForAnimation) > 1:
        # create list of files for ffmpeg
        filelist_content = "\n".join([f"file '{database+'/'+filename_h}'" for filename_h in filesForAnimation])
        with open(outdir+"/filelist.txt", "w") as f:
            f.write(filelist_content)

        command = ["ffmpeg", "-r", str(fps), "-vsync", "passthrough", "-f", "concat", "-safe", "0", "-i", outdir+"/filelist.txt", "-c:v", "libx264" , "-pix_fmt", "yuv420p", filename+".mp4"]
        with open(database+"ffmpeg_log_for_"+dataName, "w") as outfile:
            subprocess.run(command, stdout=outfile, stderr=subprocess.STDOUT, check=True)

        # Print a confirmation message
        print(f"(save_mp4): Animation file \"{filename}\".mp4 was created with {fps} fps")
    else:
        print(f"(save_mp4): WARNING: Less than 2 files found for '{dataName}', no mp4 created!")
        
class Slice2dReporter:
    def __init__(self, lattice, simulation, normal_dir = 2, position=None, domain_constraints=None, interval=None, start=None, end=None, outdir=None, show=False, cmap=None):
        self.lattice = lattice
        self.simulation = simulation
        self.interval = interval
        if self.interval < 1:
            self.interval = 1
        if start is None:
            self.start = 0
        else:
            self.start = start
        if end is None:
            self.end = np.inf
        else:
            self.end = end
        self.outdir = outdir
        self.show = show

        self.u_lim = (0, 2 * self.simulation.flow.units.characteristic_velocity_pu)
        self.p_lim = (-simulation.flow.units.characteristic_pressure_pu, simulation.flow.units.characteristic_pressure_pu)# (-1e-5, +1e-5)
        self.cmap = cmap

        if outdir is None and show is None:
            print(f"(WARNING) slice2dReporter was initialized with outdir = None and show = False... no results will be shown or saved...BRUH...")
        self.show2d_slice_reporter = Show2D(lattice, simulation.flow.solid_mask, domain_constraints, outdir, save=True if outdir is not None else False, show=show, figsize=(4, 4), dpi=dpi)
        # TODO: add secondary x- and y-axis with LU-coordinates to PLOT!, make exclusion of solid mask possible!

    def __call__(self, i, t, f):
        if self.interval is not None and (self.start+i) % self.interval == 0 and i >= self.start and i <= self.end:
            u_LU = self.lattice.u(f)
            rho_LU = self.lattice.rho(f)

            u_PU = self.simulation.flow.units.convert_velocity_to_pu(u_LU)
            p_PU = self.simulation.flow.units.convert_density_lu_to_pressure_pu(rho_LU)

            u = self.lattice.convert_to_numpy(u_PU)
            p = self.lattice.convert_to_numpy(p_PU)
            u_magnitude = np.linalg.norm(u, axis=0)

            self.show2d_slice_reporter(u_magnitude, f"u_mag(t = {t:.3f} s, step = {self.simulation.i}) noLIM",f"nolim_u_mag_i{self.simulation.i:08}_t{int(t)}", cmap=self.cmap)
            self.show2d_slice_reporter(u_magnitude, f"u_mag(t = {t:.3f} s, step = {self.simulation.i}) LIM99",f"lim99_u_mag_i{self.simulation.i:08}_t{int(t)}", vlim=(np.percentile(u_magnitude.flatten(), 1), np.percentile(u_magnitude.flatten(), 99)), cmap=self.cmap)
            self.show2d_slice_reporter(u_magnitude, f"u_mag(t = {t:.3f} s, step = {self.simulation.i}) LIM2uchar", f"lim2uchar_u_mag_i{self.simulation.i:08}_t{int(t)}", vlim=self.u_lim, cmap=self.cmap)

            self.show2d_slice_reporter(p[0], f"p (t = {t:.3f} s, step = {self.simulation.i}) noLIM", f"nolim_p_i{self.simulation.i:08}_t{int(t)}", cmap=self.cmap)
            self.show2d_slice_reporter(p[0], f"p (t = {t:.3f} s, step = {self.simulation.i}) LIM99",f"lim99_p_i{self.simulation.i:08}_t{int(t)}", vlim=(np.percentile(p[0].flatten(), 1), np.percentile(p[0].flatten(), 99)), cmap=self.cmap)
            self.show2d_slice_reporter(p[0], f"p (t = {t:.3f} s, step = {self.simulation.i}) LIMfix",f"limfix_p_i{self.simulation.i:08}_t{int(t)}", vlim=self.p_lim, cmap=self.cmap)


# ***************************************************************************************************

### ANALYSE and PROCESS PARAMETERS

# resolution and domain geometry (from given values of domain_length_x_lu, domain_length_x_pu and resolution. Two of which can be provided
if args["domain_length_x_lu"] is not None and args["domain_length_x_pu"] is not None:  # calculate resolution
    domain_length_x_lu = args["domain_length_x_lu"]
    domain_length_x_pu = args["domain_length_x_pu"]
    resolution = args["domain_length_x_lu"]/args["domain_length_x_pu"]
    
    domain_height_y_lu = args["domain_height_y_lu"]
    domain_height_y_pu = args["domain_height_y_pu"]
    domain_width_z_lu = args["domain_width_z_lu"]
    domain_width_z_pu = args["domain_width_z_pu"]
elif args["domain_length_x_lu"] is not None and args["resolution"] is not None:  # calculate PU
    domain_length_x_lu = args["domain_length_x_lu"]
    resolution = args["resolution"]
    domain_length_x_pu = args["domain_length_x_lu"] / args["resolution"]

    domain_height_y_lu = args["domain_height_y_lu"]
    domain_height_y_pu = args["domain_height_y_lu"] / args["resolution"]
    domain_width_z_lu = args["domain_width_z_lu"]
    domain_width_z_pu = args["domain_width_z_lu"] / args["resolution"]
elif args["domain_length_x_pu"] is not None and args["resolution"] is not None:  # calculate LU
    domain_length_x_pu = args["domain_length_x_pu"]
    resolution = args["resolution"]
    domain_length_x_lu = args["domain_length_x_pu"] * args["resolution"]
    
    domain_height_y_lu = args["domain_height_y_pu"] * args["resolution"]
    domain_height_y_pu = args["domain_height_y_pu"]
    domain_width_z_lu = args["domain_width_z_pu"] * args["resolution"]
    domain_width_z_pu = args["domain_width_z_pu"]
else:
    print("ERROR: domain geometry could not be determined!")
    


# flow physics
viscosity_pu = args["viscosity_pu"]
char_density_pu = args["char_density_pu"]
ma = args["ma"]

if args["keil_delta_ux_pu"] is not None and args["keil_percentage_of_inlet"] is not None:
    # calculate u_max and Reynoldsnumber
    triangle_y_pu = domain_height_y_pu * args["keil_percentage_of_inlet"] / 2
    
    keil_u_max_pu = args["keil_delta_ux_pu"] * triangle_y_pu
    re = keil_u_max_pu * domain_length_x_pu / viscosity_pu
    keil_percentage_of_inlet = args["keil_percentage_of_inlet"]
    keil_delta_ux_pu = args["keil_delta_ux_pu"]

elif args["keil_delta_ux_pu"] is not None and (args["re"] is not None or args["keil_u_max_pu"] is not None):
    # calculate percentage of inlet
    
    if args["keil_u_max_pu"] is not None:
        # calculate Reynoldsnumber
        re = args["keil_u_max_pu"] * domain_length_x_pu / viscosity_pu
        keil_u_max_pu = args["keil_u_max_pu"]
    elif args["re"] is not None:
        # calc char_velocity
        re = args["re"]
        keil_u_max_pu = re / domain_length_x_pu * viscosity_pu
    else:
        print("ERROR: could not determin Reynoldsnumber or keil_u_max_pu!")
    
    triangle_y_pu = keil_u_max_pu / args["keil_delta_ux_pu"]
    keil_delta_ux_pu = args["keil_delta_ux_pu"]
    keil_percentage_of_inlet = 2 * triangle_y_pu / domain_height_y_pu

elif (args["re"] is not None or args["keil_u_max_pu"] is not None) and args["keil_percentage_of_inlet"] is not None:
    # calculate delta_ux_pu

    if args["keil_u_max_pu"] is not None:
        # calculate Reynoldsnumber
        re = args["keil_u_max_pu"] * domain_length_x_pu / viscosity_pu
        keil_u_max_pu = args["keil_u_max_pu"]
    elif args["re"] is not None:
        # calc char_velocity
        re = args["re"]
        keil_u_max_pu = re / domain_length_x_pu * viscosity_pu
    else:
        print("ERROR: could not determin Reynoldsnumber or keil_u_max_pu!")

    keil_percentage_of_inlet = args["keil_percentage_of_inlet"]
    triangle_y_pu = domain_height_y_pu * args["keil_percentage_of_inlet"] / 2
    keil_delta_ux_pu = keil_u_max_pu / triangle_y_pu
else:
    print("ERROR: could not determin flow physics!")
        
        
        
# steps and time

t_start = 0  # TODO: add t_start argument, add t_end output
n_steps_start = 0
if args["t_target"] is not None:  # calculate steps LU
    # t_start, t_target
    t_target = args["t_target"]
    t_duration = args["t_target"] - t_start
    n_steps_duration = int(t_duration * domain_length_x_lu/domain_length_x_pu * keil_u_max_pu/(ma*1/np.sqrt(3)))
    n_steps_target = n_steps_duration + n_steps_start
elif args["n_steps_target"] is not None:
    n_steps_target = args["n_steps_target"]
    n_steps_duration = args["n_steps_target"] - n_steps_start
    t_duration = n_steps_target / (domain_length_x_lu/domain_length_x_pu * keil_u_max_pu/(ma*1/np.sqrt(3)))
    t_target = t_start + t_duration 
else:
    print("ERROR: could not determin steps and time!")



#################################################
print(f"(INFO) Trying to simulate {n_steps_target} ({n_steps_start} to {n_steps_target}) steps, representing {t_duration:.3f} seconds [PU]!")

#################################################


### SIMULATOR SETUP
print("STATUS: Simulator setup started...")
# ceate objects, link and assemble

# STENCIL
if args["stencil"] == "D2Q9":
    stencil_obj = lt.D2Q9
elif args["stencil"] == "D3Q15":
    stencil_obj = lt.D3Q15
elif args["stencil"] == "D3Q19":
    stencil_obj = lt.D3Q19
elif args["stencil"] == "D3Q27":
    stencil_obj = lt.D3Q27
else:
    print(f"ERROR: could not interpret stencil argument: {args['stencil']}...")

# precision
if args["float_dtype"] == "float32" or args["float_dtype"] == "single":
    float_dtype = torch.float32
elif args["float_dtype"] == "half" or args["float_dtype"] == "float16":
    float_dtype = torch.float16
elif args["float_dtype"] == "double" or args["float_dtype"] == "float64":
    float_dtype = torch.float64
else:
    print(f"ERROR: could not interpret float_dtype argument: {args['float_dtype']}. Using double precision.")
    float_dtype = torch.float64

# LATTICE
lattice = lt.Lattice(stencil_obj, device=torch.device(args["default_device"]), dtype=float_dtype)
if args["eqlm"]:  # use EQLM with 20% less memory usage and 2% less performance
    print("(INFO) Using Equilibrium_LessMemory (saving ~20% VRAM on GPU, but ~2% slower)")
    lattice.equilibrium = lt.QuadraticEquilibrium_LessMemory(lattice)

# DOMAIN CONSTRAINTS [PU]
print("Defining domain constraints...")
xmin_pu, ymin_pu, zmin_pu = 0, 0, 0 if lattice.D == 3 else None
xmax_pu, ymax_pu, zmax_pu = domain_length_x_pu, domain_height_y_pu, domain_width_z_pu if lattice.D == 3 else None
domain_constraints = ([xmin_pu, ymin_pu], [xmax_pu, ymax_pu]) if lattice.D == 2 else ([xmin_pu, ymin_pu, zmin_pu], [xmax_pu, ymax_pu, zmax_pu])  # Koordinatensystem abh. von der stl und deren ursprung
shape = (domain_length_x_lu, domain_height_y_lu, domain_width_z_lu) if lattice.D == 3 else (domain_length_x_lu, domain_height_y_lu)

print(f"-> Domain PU constraints = {domain_constraints}")
print(f"-> Domain LU shape = {shape}")

## FLOW Class
print("Initializing flow class...")
flow = VelocityKeilFlow(shape, re, ma, lattice, domain_constraints, domain_length_x_lu, domain_length_x_pu,
                        keil_u_max_pu, u_init=args["u_init"],
                        keil_percentage_of_inlet=keil_percentage_of_inlet,
                        keil_steigung=keil_delta_ux_pu,
                        inlet_bc=args["inlet_bc"], outlet_bc=args["outlet_bc"],
                        inlet_ramp_steps=args["inlet_ramp_steps"])

# export flow physics to file:
output_file = open(outdir+"/flow_physics_parameters.txt", "a")
output_file.write('\n{:30s}'.format("FLOW PHYSICS and units:"))
output_file.write('\n')
output_file.write('\n{:30s} {:30s}'.format("Ma", str(ma)))
output_file.write('\n{:30s} {:30s}'.format("Re", str(re)))
output_file.write('\n')
output_file.write('\n{:30s} {:30s}'.format("Relaxation Parameter LU", str(flow.units.relaxation_parameter_lu)))
output_file.write('\n{:30s} {:30s}'.format("l_char_LU", str(flow.units.characteristic_length_lu)))
output_file.write('\n{:30s} {:30s}'.format("u_char_LU", str(flow.units.characteristic_velocity_lu)))
output_file.write('\n{:30s} {:30s}'.format("viscosity_LU", str(flow.units.viscosity_lu)))
output_file.write('\n{:30s} {:30s}'.format("p_char_LU", str(flow.units.characteristic_pressure_lu)))
output_file.write('\n')
output_file.write('\n{:30s} {:30s}'.format("l_char_PU", str(flow.units.characteristic_length_pu)))
output_file.write('\n{:30s} {:30s}'.format("u_char_PU", str(flow.units.characteristic_velocity_pu)))
output_file.write('\n{:30s} {:30s}'.format("viscosity_PU", str(flow.units.viscosity_pu)))
output_file.write('\n{:30s} {:30s}'.format("p_char_PU", str(flow.units.characteristic_pressure_pu)))
output_file.write('\n')
output_file.write('\n{:30s} {:30s}'.format("grid reynolds number Re_g", str(flow.units.characteristic_velocity_lu/(lattice.stencil.cs**2 * (flow.units.relaxation_parameter_lu - 0.5)))))
output_file.write('\n{:30s} {:30s}'.format("flow through time PU", str(domain_length_x_pu/keil_u_max_pu)))
output_file.write('\n{:30s} {:30s}'.format("flow through time LU", str(flow.grid[0].shape[0]/flow.units.characteristic_velocity_lu)))
output_file.write('\n')
output_file.close()

# COLLISION
print("Initializing collision operator...")
collision_obj = None
if args["collision"].casefold() == "reg" or args["collision"].casefold() == "bgk_reg":
    collision_obj = lt.RegularizedCollision(lattice, tau=flow.units.relaxation_parameter_lu)
elif args["collision"].casefold() == "kbc":
    if lattice.D == 2:
        collision_obj = lt.KBCCollision2D(lattice, tau=flow.units.relaxation_parameter_lu)
    else:
        collision_obj = lt.KBCCollision3D(lattice, tau=flow.units.relaxation_parameter_lu)
else:  # default to bgk
    print("(!) could not determine collision, using default BGK collision operator")
    collision_obj = lt.BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)

# STREAMING
print("Initializing streaming object...")
streaming = lt.StandardStreaming(lattice)

# SIMULATION
print("Initializing simulation object...")
simulation = lt.Simulation(flow, lattice, collision_obj, streaming)
if args["t_sim_max"] > 0:
    simulation.t_max = args["t_sim_max"]

# OPTIONAL
#simulation.initialize_f_neq()

# OPTIONAL initialization process with reynolds *1/100
# to be written...

#############################################################################

### PLOT VELOCITY PROFILE AND GRADIENT OF VELOCITY PROFILE ON INLET

# PLOT VELOCITY PROFILE over central slice in XY-plane at z=int(Z/2)
fig, ax = plt.subplots(constrained_layout=True)
ux_profile_lu = flow.units.convert_velocity_to_lu(flow.u_x_keil_3D[0, :, int(shape[2] / 2)])  # (!) lattice.u gibt LU, flow.initial_solution gibt PU; flow hat ux_profile auch in PU
y_values_lu = np.arange(len(ux_profile_lu))
ux_profile_table = np.stack([y_values_lu, ux_profile_lu])
np.savetxt(outdir + f"/velocity_profile_ux_inlet.txt", ux_profile_table, header="y_value (LU) |  ux (LU)")
ax.plot(ux_profile_lu, y_values_lu, marker ="x", linestyle =":")
ax.set_xlabel("ux [LU]")
ax.set_ylabel("y [LU]")
secax = ax.secondary_yaxis('right', functions=(flow.units.convert_length_to_pu, flow.units.convert_length_to_lu))
secax.set_ylabel("y [PU]")
secax2 = ax.secondary_xaxis('top', functions=(flow.units.convert_velocity_to_pu, flow.units.convert_velocity_to_lu))
secax2.set_xlabel("ux [PU]")
plt.grid()
fig.suptitle(str(timestamp) + "\n" + args["name"] + "\n" + "velocity inlet profile")
plt.savefig(outdir+"/velocity_profile_ux_inlet.png")
if not args["cluster"]:
    plt.show()

# determine ux-velocity profile gradients...
fig, ax = plt.subplots(constrained_layout=True)
ux_profile_deltas_lu_per_lu = ux_profile_lu[1:] - ux_profile_lu[:-1]
y_values_lu = np.arange(len(ux_profile_deltas_lu_per_lu)) + 0.5
ux_profie_deltas_table = np.stack([y_values_lu, ux_profile_deltas_lu_per_lu])
np.savetxt(outdir + f"/velocity_profile_ux_deltas_inlet.txt", ux_profie_deltas_table, header="y_value (LU)  |  dux/dy (LU)")
ax.plot(ux_profile_deltas_lu_per_lu, y_values_lu, marker ="x", linestyle =":")
ax.set_xlabel("dux/dy [LU]")
ax.set_ylabel("y [LU]")
secax = ax.secondary_yaxis('right', functions=(flow.units.convert_length_to_pu, flow.units.convert_length_to_lu))
secax.set_ylabel("y [PU]")
secax2 = ax.secondary_xaxis('top', functions=(flow.units.convert_velocity_to_pu, flow.units.convert_velocity_to_lu))
secax2.set_xlabel("dux/dy [PU]")
plt.grid()
fig.suptitle(str(timestamp) + "\n" + args["name"] + "\n" + f"velocity inlet profile deltas (grad)\ndUXmaxLU = {max(ux_profile_deltas_lu_per_lu):.5f}, dUXmaxPU = {flow.units.convert_velocity_to_pu(max(ux_profile_deltas_lu_per_lu)):.5f}")
plt.savefig(outdir+"/velocity_profile_ux_deltas_inlet.png")
if not args["cluster"]:
    plt.show()

print(f"(!) maximum velocity gradient on inlet is dux/dy = {max(ux_profile_deltas_lu_per_lu)} [LU], {flow.units.convert_velocity_to_pu(max(ux_profile_deltas_lu_per_lu))} [PU]")



## CHECK INITIALISATION AND 2D DOMAIN
print(f"Initializing Show2D instances for 2d plots...")
if args["save_animations"]:
    observable_2D_plots_path = outdir_data + "/observable_2D_plots"
else:
    observable_2D_plots_path = outdir + "/observable_2D_plots"
if not os.path.isdir(observable_2D_plots_path):
    os.makedirs(observable_2D_plots_path)

show2d_observables = Show2D(lattice, flow.solid_mask, domain_constraints, outdir=observable_2D_plots_path, show=not args["cluster"], figsize=(4,4), dpi=dpi)

# plot initial u_x velocity field as 2D slice
show2d_observables(lattice.convert_to_numpy(flow.units.convert_velocity_to_pu(lattice.u(simulation.f))[0]), "u_x_INIT(t=0)", "u_x_t0")



## REPORTER
# create and append reporters
print("Initializing reporters...")

#TODO: experiment with larger interval for obs_reportes...
#TODO: measure impact of interval=1 obs_reporters...

# OBSERVABLE REPORTERS
max_u_lu_observable = lt.MaximumVelocityLU(lattice, flow, track_index=False)  # TODO: test "track_index" -> überlege sinnvolle Ausgabe bzw. plotting dafür. Grafisch? "wo"?
max_u_lu_reporter = lt.ObservableReporter(max_u_lu_observable, interval=1, out=None)
simulation.reporters.append(max_u_lu_reporter)

max_p_pu_observable = lt.MaxMinPressure(lattice, flow)
min_max_p_pu_reporter = lt.ObservableReporter(max_p_pu_observable, interval=1, out=None)
simulation.reporters.append(min_max_p_pu_reporter)


# VTK REPORTER

# 3D
if args["vtk_3D"]:
    if args["vtk_3D_t_start"] is not None:
        #print("(vtk) overwriting vtk_step_start with {}, because vtk_t_start = {}")
        vtk_3d_i_start = int(round(flow.units.convert_time_to_lu(args["vtk_3D_t_start"])))
    elif args["vtk_3D_step_start"] is not None:
        vtk_3d_i_start = int(args["vtk_3D_step_start"])
    else:
        vtk_3d_i_start = 0

    if args["vtk_3D_t_end"] is not None:
        #print("(vtk) overwriting vtk_step_end with {}, because vtk_t_end = {}")
        vtk_3d_i_end = int(flow.units.convert_time_to_lu(args["vtk_3D_t_end"]))
    elif args["vtk_3D_step_end"] is not None:
        vtk_3d_i_end = args["vtk_3D_step_end"]
    else:
        vtk_3d_i_end = n_steps_duration  # must this be target?

    if args["vtk_3D_t_interval"] is not None and args["vtk_3D_t_interval"] > 0:
        vtk_3d_interval = int(flow.units.convert_time_to_lu(args["vtk_3D_t_interval"]))
    elif args["vtk_3D_step_interval"] is not None and args["vtk_3D_step_interval"] > 0:
        vtk_3d_interval = args["vtk_3D_step_interval"]
    elif args["vtk_3D_fps"] is not None and args["vtk_3D_fps"] > 0:
        vtk_3d_interval = int(flow.units.convert_time_to_lu(1 / args["vtk_3D_fps"]))
    else:
        vtk_3d_interval = 1

    if vtk_3d_interval < 1:
        vtk_3d_interval = 1

    vtk_3d_reporter = lt.VTKReporter(lattice, flow,
                                  interval=int(vtk_3d_interval),
                                  filename_base=outdir_data + "/vtk/out",
                                  imin=vtk_3d_i_start, imax=vtk_3d_i_end)
    simulation.reporters.append(vtk_3d_reporter)

# slice2D
if args["vtk_slice2D"]:
    if args["vtk_slice2D_t_start"] is not None and args["vtk_slice2D_t_start"] > 0:
        #print("(vtk) overwriting vtk_step_start with {}, because vtk_t_start = {}")
        vtk_slice2d_i_start = int(round(flow.units.convert_time_to_lu(args["vtk_slice2D_t_start"])))
    elif args["vtk_slice2D_step_start"] is not None and args["vtk_slice2D_step_start"] > 0:
        vtk_slice2d_i_start = int(args["vtk_slice2D_step_start"])
    else:
        vtk_slice2d_i_start = 0

    if args["vtk_slice2D_t_end"] is not None and args["vtk_slice2D_t_end"] > 0:
        #print("(vtk) overwriting vtk_step_end with {}, because vtk_t_end = {}")
        vtk_slice2d_i_end = int(flow.units.convert_time_to_lu(args["vtk_slice2D_t_end"]))
    elif args["vtk_slice2D_step_end"] is not None and args["vtk_slice2D_step_end"] > 0:
        vtk_slice2d_i_end = int(args["vtk_slice2D_step_end"])
    else:
        vtk_slice2d_i_end = n_steps_duration  # Q: must this be target?

    if args["vtk_slice2D_t_interval"] is not None and args["vtk_slice2D_t_interval"] > 0:
        vtk_slice2d_interval = int(flow.units.convert_time_to_lu(args["vtk_slice2D_t_interval"]))
    elif args["vtk_slice2D_step_interval"] is not None and args["vtk_slice2D_step_interval"] > 0:
        vtk_slice2d_interval = int(args["vtk_slice2D_step_interval"])
    elif args["vtk_slice2D_fps"] is not None and args["vtk_slice2D_fps"] > 0:
        vtk_slice2d_interval = int(flow.units.convert_time_to_lu(1 / args["vtk_slice2D_fps"]))
    else:
        vtk_slice2d_interval = 1

    if vtk_slice2d_interval < 1:
        vtk_slice2d_interval = 1

    vtk_domainSlice_reporter = lt.VTKsliceReporter(lattice, flow,
                                                   interval=int(vtk_slice2d_interval),
                                                   filename_base=outdir_data + "/vtk/slice_domain/slice_domain",
                                                   sliceXY=([0,shape[0]-1],[0,shape[1]-1]),
                                                   sliceZ=int(shape[2]/2),
                                                   imin=vtk_slice2d_i_start, imax=vtk_slice2d_i_end)
    simulation.reporters.append(vtk_domainSlice_reporter)

# TODO add vtk_slice_interval reporter...
# # vtk_slices for specific intervals
#     i_intervals_vtk_domain_slice = np.array([[0,1000],
#                                              [5000,5050],
#                                              [20000,20050],
#                                              [40000,40050],
#                                              [60000,60050],
#                                              [80000,80050],
#                                              [99950,100000]])
#     vtk_domainSliceInterval_reporters = [None] * i_intervals_vtk_domain_slice.shape[0]
#     for interval_min_max in range(i_intervals_vtk_domain_slice.shape[0]):
#         print(f"(INFO): Adding vtk_slice_domain_2D reporter for interval [{i_intervals_vtk_domain_slice[interval_min_max,0]}, {i_intervals_vtk_domain_slice[interval_min_max,1]}]")
#         vtk_domainSliceInterval_reporters[interval_min_max] = lt.VTKsliceReporter(lattice, flow,
#                                                        interval=1,
#                                                        filename_base=outdir_data + "/vtk/slice_domain_intervals/slice_domain_intervals",
#                                                        sliceXY=([0,flow.shape[0]-1], [0,flow.shape[1]-1]),
#                                                        sliceZ=outlet_sliceZ,
#                                                        imin=i_intervals_vtk_domain_slice[interval_min_max,0], imax=i_intervals_vtk_domain_slice[interval_min_max,1])
#         simulation.reporters.append(vtk_domainSliceInterval_reporters[interval_min_max])

# WATCHDOG-REPORTER (reports runtime, estimated end etc.)
if args["watchdog"]:
    if args["watchdog_interval"] < 1:
        baguette = int(n_steps_duration/100)
    else:
        baguette = int(args["watchdog_interval"])
    watchdog_reporter = lt.Watchdog(lattice, flow, simulation, interval=baguette, i_start=n_steps_start, i_target=n_steps_target, t_max=simulation.t_max, filebase=outdir+"/watchdog", show=not args["cluster"])
    simulation.reporters.append(watchdog_reporter)

# NAN REPORTER (stops sim if NaN is detected)
if args["nan_reporter"]:
    nan_reporter = lt.NaNReporter(flow, lattice, n_steps_target, t_target, interval=args["nan_reporter_interval"], simulation=simulation, vtk_dir=outdir_data+"/vtk", vtk=True, outdir=outdir)  # omitting outdir leads to no extra file with coordinates being created. With a resolution of >100.000.000 Gridpoints, torch gets confused otherwise...
    simulation.reporters.append(nan_reporter)

# HIGH MA REPORTER (reports high Ma positions (Ma>0.3))
if args["high_ma_reporter"]:
    high_ma_reporter_path = outdir+"/HighMaReporter"
    # if not os.path.exists(high_ma_reporter_path):
    #     os.makedirs(high_ma_reporter_path)
    high_ma_reporter = lt.HighMaReporter(flow, lattice, n_steps_target, t_target, interval=args["high_ma_reporter_interval"], simulation=simulation, outdir=high_ma_reporter_path, vtk_dir=outdir_data+"/vtk/HighMa", stop_simulation=False, vtk_highma_points=True)  # stop_simulation overwrites vtk output of HighMaReporter with False
    simulation.reporters.append(high_ma_reporter)

#TODO: set watchdog- and high_ma_reporter intervals if SET in arguments!

# 2D SLICE PNG reporter:
if args["save_animations"]:
    if args["animations_number_of_frames"] > 0:  # number of 2D frames to take for animations
        interval = int(n_steps_target/args["animations_number_of_frames"])
    elif args["animations_fps_pu"] > 0:  # if no number of frames given, calculate it from fps
        number_of_frames = t_target * args["animations_fps_pu"]
        interval = int(n_steps_target/number_of_frames)
    else:  #neither number of frames, nor fps are given, take 1000 frames...
        interval = int(n_steps_target / 1000)

    slice2dReporter = Slice2dReporter(lattice, simulation, domain_constraints=domain_constraints, interval=interval, start=0, outdir=observable_2D_plots_path, show = False)
    simulation.reporters.append(slice2dReporter)



## WRITE PARAMETERS to file in outdir
# TODO: write parameters to file (ähnlich zu mdout.mdp)
#  siehe auch oben, args wird rausgeschrieben als "input_parameters". Die konkrete ÄNDERUNG aller inputs für die Sim müsste ich dann nochmal hier händisch schreiben



## LOAD CHECKPOINT FILE
# TODO: load checkpoint file and adjust sim.i


# (TEMP) FLOW THROUGH TIME
print(f"INFO: flow through time (time a particle takes to travel from input to output unobstructed at u_char): T_ft_PU = {domain_length_x_pu/flow.units.characteristic_velocity_pu} s")
print(f"(debug) flow through time (calc. as grid.shape[0]/u_char_lu) T_ft_LU = {flow.grid[0].shape[0]/flow.units.characteristic_velocity_lu} steps")
print(f"(debug) flow through time (calc. as convert_PU_to_LU(T_ft_PU) T_ft_LU = {flow.units.convert_time_to_lu(domain_length_x_pu/flow.units.characteristic_velocity_pu)} steps")



### RUN SIMULATION
#n_steps = n_stop_target - n_start
t_start = time()

print(f"\n\n***** SIMULATION STARTED AT {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} *****\n")
mlups = simulation.step(n_steps_duration)

t_end = time()
runtime = t_end-t_start

# PRINT SOME STATS TO STDOUT:
print(f"\n***** SIMULATION FINISHED AT {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} *****\n")
print("MLUPS:", mlups)
print(f"runtime: {runtime:.3f} seconds (= {round(runtime/60, 2)} min = {round(runtime/3600, 2)} h)")
print(f"\nsimulated PU-Time: {flow.units.convert_time_to_pu(simulation.i):.3f} seconds")
print("simulated number of steps:", simulation.i)
print(f"Domain (L,H,B) = {flow.shape}")
print(f"number of gridpoints = {flow.shape[0]*flow.shape[1]*flow.shape[2] if len(flow.shape) else flow.shape[0]*flow.shape[1]}")
# TODO: PU_time (target, reached), n_steps (target, reached), gridpoints, domain (GP³)
grid_reynolds_number = flow.units.characteristic_velocity_lu/(lattice.stencil.cs**2 * (flow.units.relaxation_parameter_lu - 0.5))  # RE_grid as a measure for free flow resolution (should be ~O(10) )
print(f"-> Grid Reynolds number Re_grid = {grid_reynolds_number:.3f}")

print("\n")
# TODO: change output to f-string
print("\n*** HARDWARE UTILIZATION ***\n")
print(f"current GPU VRAM (MB) usage: {torch.cuda.memory_allocated(device=args['default_device'])/1024/1024:.3f}")
print(f"max. GPU VRAM (MB) usage: {torch.cuda.max_memory_allocated(device=args['default_device'])/1024/1024:.3f}")

[cpuLoad1,cpuLoad5,cpuLoad15] = [x / psutil.cpu_count() * 100 for x in psutil.getloadavg()]
print("CPU LOAD AVG.-% over last 1 min, 5 min, 15 min; ", round(cpuLoad1,2), round(cpuLoad5,2), round(cpuLoad15,2))

ram = psutil.virtual_memory()
print("Current total (CPU) RAM usage [MB]: " + str(round(ram.used/(1024*1024),2)) + " of " + str(round(ram.total/(1024*1024),2)) + " MB")
print("maximum total (CPU) RAM usage ('MaxRSS') [MB]: " + str(round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, 2)) + " MB")

print("\n")


### export stats
output_file = open(outdir+"/stats.txt", "a")
output_file.write("DATA for "+timestamp)
output_file.write("\n\n###   SIM-STATS  ###")
output_file.write(f"\nruntime: {runtime:.3f} seconds (= {round(runtime/60, 2)} min = {round(runtime/3600, 2)} h)")
output_file.write("\nMLUPS = "+str(mlups))
output_file.write("\n")
output_file.write("\nVRAM_current [MB] = " + str(torch.cuda.memory_allocated(lattice.device)/1024/1024))
output_file.write("\nVRAM_peak [MB] = " + str(torch.cuda.max_memory_allocated(lattice.device)/1024/1024))
output_file.write("\n")
output_file.write("\nCPU load % avg. over last 1, 5, 15 min: " + str(round(cpuLoad1, 2)) + " %, " + str(round(cpuLoad5, 2)) + " %, " + str(round(cpuLoad15, 2)) + " %")
output_file.write("\nCurrent total RAM usage [MB]: " + str(round(ram.used/(1024*1024),2)) + " of " + str(round(ram.total/(1024*1024),2)) + " MB")
output_file.write("\nmaximum total RAM usage ('MaxRSS') [MB]: " + str(round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, 2)) + " MB")
output_file.close()

### export CUDA-summary
output_file = open(outdir+"/GPU_memory_summary.txt", "a")
output_file.write("DATA for "+timestamp+"\n\n")
output_file.write(torch.cuda.memory_summary(device=lattice.device))
output_file.close()


### WRITE CHECKPOINT
# TODO: write checkpointfile, if write_cpt=True and save sim.i to it / should be on correct device! or CPU to be save

### POSTPROCESSONG: PROCESS and export observables
# TODO: Observable post-processing, while flow and sim data is still available

### OUTPUT RESULTS
# TODO: output results to human-readable file AND to copy-able file (or csv)


## 2D png-slices and mp4-movie
if args["save_animations"]:  # takes images from slice2dReporter and created mp4
    t0 = time()
    print(f"(INFO) creating animations from slice2dRepoter Data...")
    os.makedirs(outdir_data+"/animations")

    if args["animations_fps_mp4"] > 0:
        fps = int(args["animations_fps_mp4"])
    else:
        fps = 3

    save_mp4(outdir_data+"/animations/lim99_u_mag", observable_2D_plots_path, "lim99_u_mag", fps=fps)
    save_mp4(outdir_data+"/animations/lim2uchar_u_mag", observable_2D_plots_path, "lim2uchar_u_mag", fps=fps)
    save_mp4(outdir_data+"/animations/nolim_u_mag", observable_2D_plots_path, "nolim_u_mag", fps=fps)

    save_mp4(outdir_data+"/animations/lim99_p", observable_2D_plots_path, "lim99_p", fps=fps)
    save_mp4(outdir_data+"/animations/limfix_p", observable_2D_plots_path, "limfix_p", fps=fps)
    save_mp4(outdir_data+"/animations/nolim_p", observable_2D_plots_path, "nolim_p", fps=fps)
    t1 = time()
    print(f"Creating animations from slice2dRepoter Data took {floor((t1 - t0) / 60):02d}:{floor((t1- t0) % 60):02d} [mm:ss]")
else:  # only plots final u_mag and p fields
    print(f"(INFO) plotting final u_mag and p fields...")
    t0 = time()
    t = flow.units.convert_time_to_pu(simulation.i)
    u_LU = lattice.u(simulation.f)
    rho_LU = lattice.rho(simulation.f)

    u_PU = flow.units.convert_velocity_to_pu(u_LU)
    p_PU = flow.units.convert_density_lu_to_pressure_pu(rho_LU)

    u = lattice.convert_to_numpy(u_PU)
    p = lattice.convert_to_numpy(p_PU)
    u_magnitude = np.linalg.norm(u, axis=0)

    u_lim = (0, 2 * simulation.flow.units.characteristic_velocity_pu)
    p_lim = (-1e-5, +1e-5)

    show2d_observables(u_magnitude, f"u_mag(t = {t:.3f} s, step = {simulation.i}) noLIM",
                               f"nolim_u_mag_i{simulation.i:08}_t{int(t)}")
    show2d_observables(u_magnitude, f"u_mag(t = {t:.3f} s, step = {simulation.i}) LIM99",
                               f"lim99_u_mag_i{simulation.i:08}_t{int(t)}",
                               vlim=(np.percentile(u_magnitude.flatten(), 1), np.percentile(u_magnitude.flatten(), 99)))
    show2d_observables(u_magnitude, f"u_mag(t = {t:.3f} s, step = {simulation.i}) LIM2CHAR",
                               f"lim2uchar_u_mag_i{simulation.i:08}_t{int(t)}",
                               vlim=u_lim)
    t1 = time()
    if not args["cluster"]:
        print("(INFO) Waiting for 6.5 seconds to avoid 'HTTP Error 429: Too Many Requests'...")
        sleep(max(7 - (t1 - t0), 0))
    show2d_observables(p[0], f"p (t = {t:.3f} s, step = {simulation.i}) noLIM",
                               f"nolim_p_i{simulation.i:08}_t{int(t)}")
    show2d_observables(p[0], f"p (t = {t:.3f} s, step = {simulation.i}) LIM99",
                               f"lim99_p_i{simulation.i:08}_t{int(t)}",
                               vlim=(np.percentile(p[0].flatten(), 1), np.percentile(p[0].flatten(), 99)))
    show2d_observables(p[0], f"p (t = {t:.3f} s, step = {simulation.i}) LIMFIX",
                               f"limfix_p_i{simulation.i:08}_t{int(t)}",
                               vlim=p_lim)
####

## PLOTTING of max.Ma, max.u_mag, min/max p over time:

max_u_lu = np.array(max_u_lu_reporter.out)
np.savetxt(outdir + f"/max_u_lu_timeseries.txt", max_u_lu, header="stepLU  |  timePU  |  u_mag_max_LU")

# PLOT max. Ma in domain over time...
fig, ax = plt.subplots(constrained_layout=True)
ax.plot(max_u_lu[:, 1], max_u_lu[:, 2]/lattice.convert_to_numpy(lattice.cs))
ax.set_xlabel("physical time / s")
ax.set_ylabel("maximum Ma")
ax.set_ylim([0,0.3])
secax = ax.secondary_xaxis('top', functions=(flow.units.convert_time_to_lu, flow.units.convert_time_to_pu))
secax.set_xlabel("timesteps (simulation time / LU)")
fig.suptitle(str(timestamp) + "\n" + args["name"] + "\n" + "max. Mach")
plt.savefig(outdir+"/max_Ma.png")
if not args["cluster"]:
    plt.show()

# PLOT max. u_mag for abs. anaylsis (y_limits from first part of sim)
fig, ax = plt.subplots(constrained_layout=True)
ax.plot(max_u_lu[:, 1], flow.units.convert_velocity_to_pu(max_u_lu[:, 2]))
ax.set_xlabel("physical time / s")
ax.set_ylabel("maximum momentary velocity magnitude (PU)")
y_lim_50_first = flow.units.convert_velocity_to_pu(max_u_lu[:int(max_u_lu.shape[0]/1.3), 2].max())  # max u_mag of first part of the data (excludes crash, if present, includes settling period)
ax.set_ylim([0, y_lim_50_first*1.1])  # show 10% more than u_mag_max
secax = ax.secondary_xaxis('top', functions=(flow.units.convert_time_to_lu, flow.units.convert_time_to_pu))
secax.set_xlabel("timesteps (simulation time / LU)")
fig.suptitle(str(timestamp) + "\n" + args["name"] + "\n" + "max. u_mag (ylim from 0-0.75 T)")
plt.savefig(outdir+"/max_u_mag_lim_start.png")
if not args["cluster"]:
    plt.show()

# ...(y_limits from last part of sim)
fig, ax = plt.subplots(constrained_layout=True)
ax.plot(max_u_lu[:, 1], flow.units.convert_velocity_to_pu(max_u_lu[:, 2]))
ax.set_xlabel("physical time / s")
ax.set_ylabel("maximum momentary velocity magnitude (PU)")
y_lim_50_second = flow.units.convert_velocity_to_pu(max_u_lu[int(max_u_lu.shape[0]*0.25):int(max_u_lu.shape[0]*0.95), 2].max())  # max u_mag of first part of the data (excludes crash, if present, includes settling period)
ax.set_ylim([0, y_lim_50_second*1.1 if abs(y_lim_50_second)<1000 else 1])  # show 10% more than u_mag_max
secax = ax.secondary_xaxis('top', functions=(flow.units.convert_time_to_lu, flow.units.convert_time_to_pu))
secax.set_xlabel("timesteps (simulation time / LU)")
fig.suptitle(str(timestamp) + "\n" + args["name"] + "\n" + "max. u_mag (ylim from 0.25-1 T)")
plt.savefig(outdir+"/max_u_mag_lim_end.png")
if not args["cluster"]:
    plt.show()

# PLOT max/min p for abs. analysis
min_max_p_pu = np.array(min_max_p_pu_reporter.out)
np.savetxt(outdir + f"/min_max_p_pu_timeseries.txt", min_max_p_pu, header="stepLU  |  timePU  |  p_min_PU  |  p_max_PU")
# y_lim from first part of sim...
fig, ax = plt.subplots(constrained_layout=True)
ax.plot(min_max_p_pu[:, 1], min_max_p_pu[:, 2], label='min. Pressure')
ax.plot(min_max_p_pu[:, 1], min_max_p_pu[:, 3], label='max. Pressure')
ax.set_xlabel("physical time / s")
ax.set_ylabel("min. and max. momentary pressure (PU)")
y_lim_50_min = min_max_p_pu[:int(min_max_p_pu.shape[0]/1.3), 2].min()
y_lim_50_max = min_max_p_pu[:int(min_max_p_pu.shape[0]/1.3), 3].max()
ax.set_ylim([y_lim_50_min-0.1*abs(y_lim_50_min), y_lim_50_max+0.1*abs(y_lim_50_max)])  # show 10% more above and below
secax = ax.secondary_xaxis('top', functions=(flow.units.convert_time_to_lu, flow.units.convert_time_to_pu))
secax.set_xlabel("timesteps (simulation time / LU)")
ax.legend()
fig.suptitle(str(timestamp) + "\n" + args["name"] + "\n" + "max./min. p (ylim from 0-0.75 T)")
plt.savefig(outdir+"/min_max_p_lim_start.png")
if not args["cluster"]:
    plt.show()

# y_lim from last part half of sim...
fig, ax = plt.subplots(constrained_layout=True)
ax.plot(min_max_p_pu[:, 1], min_max_p_pu[:, 2], label='min. Pressure')
ax.plot(min_max_p_pu[:, 1], min_max_p_pu[:, 3], label='max. Pressure')
ax.set_xlabel("physical time / s")
ax.set_ylabel("min. and max. momentary pressure (PU)")
y_lim_50_min_second = min_max_p_pu[int(min_max_p_pu.shape[0]*0.25):int(min_max_p_pu.shape[0]*0.95), 2].min()
y_lim_50_max_second = min_max_p_pu[int(min_max_p_pu.shape[0]*0.25):int(min_max_p_pu.shape[0]*0.95), 3].max()
ax.set_ylim([y_lim_50_min_second-0.1*abs(y_lim_50_min_second), y_lim_50_max_second+0.1*abs(y_lim_50_max_second)])  # show 10% more above and below
secax = ax.secondary_xaxis('top', functions=(flow.units.convert_time_to_lu, flow.units.convert_time_to_pu))
secax.set_xlabel("timesteps (simulation time / LU)")
ax.legend()
fig.suptitle(str(timestamp) + "\n" + args["name"] + "\n" + "max./min. p (ylim from 0.25-1 T)")
plt.savefig(outdir+"/min_max_p_lim_end.png")
if not args["cluster"]:
    plt.show()

# PRESSURE and VELOCITY at points
# only for ground and house sims...
# for i in range(len(x_positions_lu)):
#     fig_pressure, ax_pressure = plt.subplots(constrained_layout=True)
#     fig_velocity, ax_velocity = plt.subplots(constrained_layout=True)
#     for j in range(len(y_positions_lu)):
#         data = np.array(up_point_reporters[i][j].out)
#         np.savetxt(outdir + f"/PU_point_report/up_xyz{up_point_reporters[i][j].index_lu}.txt", data, header="stepLU  |  timePU  |  p  |  ux  | uy  |  uz  (obs. in PU)")
#         ax_pressure.plot(data[:,1], data[:,2], label=f"p at {up_point_reporters[i][j].index_lu}")
#         ax_velocity.plot(data[:, 1], np.sqrt(np.square(data[:, 3]) + np.square(data[:, 4]) + np.square(data[:, 5])), label=f"u_mag at {up_point_reporters[i][j].index_lu}")
#     ax_pressure.set_xlabel("physical time / s")
#     ax_velocity.set_xlabel("physical time / s")
#     ax_pressure.set_ylabel("p_PU")
#     ax_velocity.set_ylabel("u_mag_PU")
#
#     y_max_velocity = flow.units.convert_velocity_to_pu(max_u_lu[:int(max_u_lu.shape[0] / 1.3),2].max())  # max u_mag of first 50% of the data (excludes crash, if present, includes settling period)
#     ax_velocity.set_ylim([0, y_max_velocity * 1.1])  # show 10% more than u_mag_max
#     y_min_pressure = min_max_p_pu[:int(min_max_p_pu.shape[0] / 1.3), 2].min()
#     y_max_pressure = min_max_p_pu[:int(min_max_p_pu.shape[0] / 1.3), 3].max()
#     ax_pressure.set_ylim([y_min_pressure - 0.1 * abs(y_min_pressure),
#                  y_max_pressure + 0.1 * abs(y_max_pressure)])  # show 10% more above and below
#
#     ax_velocity.axhline(y=flow.units.characteristic_velocity_pu, color='tab:green', ls=":", label="characteristic velocity")
#     ax_velocity.axhline(y=flow.units.convert_velocity_to_pu(lattice.convert_to_numpy(lattice.cs) * 0.3), color='tab:red', ls=":",
#                label="Ma = 0.3")
#
#     secax_u = ax_velocity.secondary_xaxis('top', functions=(flow.units.convert_time_to_lu, flow.units.convert_time_to_pu))
#     secax_u.set_xlabel("timesteps (simulation time / LU)")
#     secax_p = ax_pressure.secondary_xaxis('top', functions=(flow.units.convert_time_to_lu, flow.units.convert_time_to_pu))
#     secax_p.set_xlabel("timesteps (simulation time / LU)")
#
#     fig_pressure.suptitle(str(timestamp) + "\n" + name + " \n" + f"p(y,t) at x = {up_point_reporters[i][j].index_lu[0]}")
#     ax_pressure.legend()
#     fig_pressure.savefig(outdir + f"/PU_point_report/p_xlu{up_point_reporters[i][j].index_lu[0]:03d}.png")
#     if not cluster:
#         fig_pressure.show()
#
#     fig_velocity.suptitle(
#         str(timestamp) + "\n" + name + " \n" + f"u_mag(y,t) at x = {up_point_reporters[i][j].index_lu[0]}")
#     ax_velocity.legend()
#     fig_velocity.savefig(outdir + f"/PU_point_report/u_mag_xlu{up_point_reporters[i][j].index_lu[0]:03d}.png")
#     if not cluster:
#         fig_velocity.show()

print("\nmaximum total (CPU) RAM usage ('MaxRSS') (including optional PNG and GIF post-processing [MB]: " + str(round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, 2)) + " MB")

## END OF SCRIPT
print(f"\n♬ THE END ♬")
sys.stdout = old_stdout

