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
import subprocess  # for mp4 export...

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
parser.add_argument("--float_dtype", default="float64", choices=["float32", "float64", "single", "double", "half"], help="data type for floating point calculations in torch")
parser.add_argument("--t_sim_max", default=(72*60*60), type=float, help="max. walltime [s] to simulate, default is 72 h for cluster use. sim stops at 0.99*t_max_sim")  # andere max.Zeit? wie lange braucht das "drum rum"? kann cih simulation auch die scho vergangene Zeit übergeben? dann kann ich mit nem größeren Wert rechnen und sim ist variabel darin, wie viel Zeit es noch hat

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

parser.add_argument("--vtk_slice_inlet", action='store_true', help="toggle vtk-output of 2D slice for inlet-interaction analysis to outdir_data, if set True (1)")
parser.add_argument("--vtk_slice_outlet", action='store_true', help="toggle vtk-output of 2D slice for outlet-interaction analysis to outdir_data, if set True (1)")
parser.add_argument("--vtk_slice_x", default=10, type=int, help="x_length of 2D vtk slice")
parser.add_argument("--vtk_slice_y", default=5, type=int, help="y_height of 2D vtk slice")
parser.add_argument("--vtk_slice_z", default=0, type=int, help="z location of 2D slice; NOT IMPLEMENTED")
parser.add_argument("--vtk_slice_intervals", action='store_true', help="toggle vtk-output of 2D slice for specific intervals (hardcoded, see below)")

# SURVEILANCE reporters
parser.add_argument("--nan_reporter", action='store_true', help="stop simulation if NaN is detected in f field")
parser.add_argument("--nan_reporter_interval", default=100, type=int, help="interval in which the NaN reporter checks f for NaN")
parser.add_argument("--high_ma_reporter", action='store_true', help="stop simulation if Ma > 0.3 is detected in u field")
parser.add_argument("--high_ma_reporter_interval", default=100, type=int, help="interval in which the HighMa reporter checks for Ma>0.3")
parser.add_argument("--watchdog", action='store_true', help="report progress, ETA and warn, if Sim is estimated to run longer than t_max (~72 h)")
parser.add_argument("--watchdog_interval", default=0, type=int, help="interval in which the watchdog reporter reports. 0 sets 100 reports per simulation")

parser.add_argument("--from_cpt", action='store_true', help="start from checkpoint. (!) provide --cpt_file path")
parser.add_argument("--cpt_file", default=None, help="path and name of cpt_file to use if --from_cpt=True")
parser.add_argument("--sim_i", default=0, type=int, help="step index of last checkpoints-step to start from for time-indexing of observables etc.")
parser.add_argument("--write_cpt", action='store_true', help="write checkpoint after finishing simulation")

# flow physics
parser.add_argument("--re", default=200, type=float, help="Reynolds number")
parser.add_argument("--ma", default=0.05, type=float, help="Mach number (should stay < 0.3, and < 0.1 for highest accuracy. low Ma can lead to instability because of round of errors ")
parser.add_argument("--viscosity_pu", default=14.852989758837 * 10**(-6), type=float, help="kinematic fluid viscosity in PU. Default is air at ~14.853e-6 (at 15°C, 1atm)")
parser.add_argument("--char_density_pu", default=1.2250, type=float, help="density, default is air at ~1.2250 at 15°C, 1atm")  # ist das so korrekt? - von Martin Kiemank übernommen
parser.add_argument("--u_init", default=0, type=int, choices=[0, 1, 2], help="0: initial velocity zero, 1: velocity one uniform, 2: velocity profile") # könnte ich noch auf Philipp und mich anpassen...und uniform durch komplett WSP ersetzen
# char velocity PU will be calculated from Re, viscosity and char_length!

# solver settings
parser.add_argument("--n_steps", default=100000, type=int, help="number of steps to simulate, overwritten by t_target, if t_target is >0, end of sim will be step_start+n_steps")
parser.add_argument("--t_target", default=0, type=float, help="time in PU to simulate, t_start will be calculated by PU/LU-conversion of step_start")
parser.add_argument("--step_start", default=0, type=int, help="stepnumber to start at. Useful if sim. is started from a checkpoint and sim-data should be concatenated later on")
parser.add_argument("--collision", default="bgk", type=str, choices=["kbc", "bgk", "reg", 'reg', "bgk_reg", 'kbc', 'bgk', 'bgk_reg'], help="collision operator (bgk, kbc, reg)")
parser.add_argument("--dim", default=3, type=int, help="dimensions: 2D (2), oder 3D (3, default)")
parser.add_argument("--stencil", default="D3Q27", choices=['D2Q9', 'D3Q15', 'D3Q19', 'D3Q27'], help="stencil (D2Q9, D3Q27, D3Q19, D3Q15), dimensions will be infered from D")
parser.add_argument("--eqlm", action="store_true", help="use Equilibium LessMemory to save ~20% on GPU VRAM, sacrificing ~2% performance")

# house and domain geometry
parser.add_argument("--house_length_lu", default=10, type=int, help="house length in LU")  # characteristic length LU, in flow direction
parser.add_argument("--ground_height_lu", default=0.5, type=float, help="ground height in LU, height ZERO, in absolute LU coordinates relative to (?) coordinate system")
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

parser.add_argument("--wsp_shift_up_lu", default=0, type=float, help="how many LU to shift the y_0 and y_ref of u_inlet profile upwards (to mitigate interaction of ground_BC and inlet); shifts whole profile including y0 and y_ref" )
parser.add_argument("--wsp_y0_lu", default=0, type=float, help="zero-height for wind speed profile below which U(y<y0)=0; only shifts y0 up, thereby squeezes profile between y_0 and y_ref 8y_ref not altered)")
parser.add_argument("--wsp_alpha", default=0.25, type=float, help="exponent for wind speed profile power law.")

parser.add_argument("--combine_solids", action='store_true', help="combine all solids (house and ground) into one object for easier prototyping")
parser.add_argument("--no_house", action='store_true', help="if TRUE, removes house from simulation, for debugging of house-independent aspects")

# boundary algorithms
parser.add_argument("--inlet_bc", default="eqin", help="inlet boundary condition: EQin, NEX, SEI, rampEQin")
parser.add_argument("--inlet_ramp_steps", default=1, type=int, help="step number over which the velocity of ramped EquilibriumInlet is ramped to 100%")
parser.add_argument("--outlet_bc", default="eqoutp", help="outlet boundary condition: EQoutP, EQoutU")
parser.add_argument("--ground_bc", default="fwbb", help="ground boundary condition: fwbb, hwbb, ibb")
parser.add_argument("--house_bc", default="fwbb", help="house boundary condition: fwbb, hwbb, ibb")
parser.add_argument("--top_bc", default="zgo", help="top boundary condition: zgo, eq")

# plotting and output
parser.add_argument("--plot_intersection_info", action='store_true', help="plot intersection info to outdir to debug solid-boundary problems")
parser.add_argument("--verbose", action='store_true', help="display more information in console (for example about neighbour search)")
parser.add_argument("--save_animations", action='store_true', help="create and save animations and pngs of u and p fields")
parser.add_argument("--animations_number_of_frames", default=0, type=int, help="number of frames to take over the course of the simulation every t_target/#frames time units, overwrites animations_fps!")
parser.add_argument("--animations_fps_pu", default=0, type=int, help="number of frames per second PU for 2D animations (GIFs). Not the fps for the GIF, but the rate at which frames are taken from simulation (relative to it's simulated PU-time)")
parser.add_argument("--animations_fps_mp4", default=0, type=int, help="number of frames per second PU for 2D animations (GIFs). Actual fps of the resulting GIF. (Not the fps at which frames are taken from simulation!")
parser.add_argument("--plot_sbd_2d", action='store_true', help="plot 2d_slices of boundary masks, solid_boundary f_indices etc.")


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
parser.add_argument("--solid_boundary_data_path", default=os.path.join(os.getcwd(), 'solid_boundary_data'), type=str, help="")  # DAS BRAUCH ICH...
parser.add_argument("--no_store_solid_boundary_data", action='store_true', help="") # ob coll_data gespeichert wird, oder nicht... -> ohne, wirds zwar verwendet, aber nicht gespeichert
# parser.add_argument("--double_precision", action='store_true', help="") # ist bei mir als float_dtype hinterlegt (s.o.)
parser.add_argument("--recalc", action='store_true', help="recalculate solid_boundary_data") # DAS BRAUCHE ICH AUCH
# parser.add_argument("--notree", action='store_true', help="") # NOT USED
# parser.add_argument("--vmax", default=1, type=float, help="note: estimate!")  # not: estimate! / das ist char_velocity PU
# parser.add_argument("--saturation", default=0.5, type=float, help="canopy partial saturation")  # not: estimate! / NOT USED
# parser.add_argument("--cut_z", default=0, type=float, help="cut at z=") # NOT USED

args = vars(parser.parse_args())

# get parameters from args[] dict:
name, default_device, float_dtype, t_sim_max, cluster, outdir, outdir_data, vtk, vtk_fps, vtk_interval, vtk_step_start, \
    nan_reporter, from_cpt, sim_i, write_cpt, re, ma, viscosity_pu, char_density_pu, u_init, n_steps, t_target, \
    step_start, collision, dim, stencil, eqlm, house_length_lu, house_length_pu, house_width_pu, roof_angle, \
    eg_height_pu, roof_height_pu, overhang_pu, domain_length_pu, domain_width_pu, domain_height_pu, inlet_bc, outlet_bc, \
    ground_bc, house_bc, top_bc, combine_solids, verbose = \
    [args[_] for _ in ["name", "default_device", "float_dtype", "t_sim_max", "cluster", "outdir", "outdir_data",
                       "vtk_3D", "vtk_3D_fps", "vtk_3D_step_interval", "vtk_3D_step_start", "nan_reporter", "from_cpt", "sim_i",
                       "write_cpt", "re", "ma", "viscosity_pu", "char_density_pu", "u_init", "n_steps", "t_target",
                       "step_start", "collision", "dim", "stencil", "eqlm", "house_length_lu", "house_length_pu",
                       "house_width_pu", "roof_angle", "eg_height_pu", "roof_height_pu", "overhang_pu", "domain_length_pu",
                       "domain_width_pu", "domain_height_pu", "inlet_bc", "outlet_bc", "ground_bc", "house_bc",
                       "top_bc", "combine_solids", "verbose"]]

# CREATE timestamp, sim-ID, outdir and outdir_data
timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
sim_id = str(timestamp) + "-" + name
os.makedirs(outdir+"/"+sim_id)
print(f"Outdir/simID = {outdir}/{sim_id}")
if outdir_data is None:
    outdir_data = outdir
outdir = outdir+"/"+sim_id  # adding individal sim-ID to outdir path to get individual DIR per simulation
outdir_data = outdir_data+"/"+sim_id
if (vtk or args["save_animations"]) and not os.path.exists(outdir_data):
    os.makedirs(outdir_data)
    print(f"Outdir_DATA/simID = {outdir}/{sim_id}")
print(f"Outdir/simID = {outdir}/{sim_id}")
print(f"Input arguments: {args}")

# save input parameters to file
output_file = open(outdir+"/input_parameters.txt", "a")
for key in args:
    output_file.write('{:30s} {:30s}\n'.format(str(key), str(args[key])))
output_file.close()

if cluster:
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

### HAUS Flow

# u_init (Geschw.feld der Init)
# SEI: K_Factor, L, N

#pspelt ?
# shape ist LU
# domain_constraints sind PU
# grid sind die PU-Koordinaten der Gridpunte -> das ist bei mir auch so! :)
# res ist der LU/PU Umrechungsfaktor
# depth?, debug?, fwbb, house?, parallel?, cluster?
# initial_solution returnt tensoren p und u

## Boundary Masken/Positionen und Trennung/Sortierung
# - inlet, outlet, top, bottom/ground, house
# (!) house-mask jetzt über Philipps kram bzw. resultiert die solid_boundary_data!
# (?) möchte ich die FWBB, HWBB auch auf "solid_boundary_data" anpassen?
# (!) behandle Ecken/Kanten sorgfältig, insb. auch bzgl. der nicht-lokalen Boundaries
# (!) behandle zwei "nahe", berührende oder überlagernde Solid-Boundaries!
#   - IDEE: solid_maske und solid_surface Maske? -> muss man das definieren,
#   ...oder kann man die solid_masken der Boundaries mit den f_index Listen vergleichen und schauen,
#   ...ob die f_index[i] in einem anderen solid liegt? Dann kann man die entsprechend aus der Liste für bounce und force rausnehmen

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


def save_gif(filename: str = "./animation",
             database: str = None,
             dataName: str = None,
             fps: int = 20,
             loop: int = 0):
    """
    Description:
    The save_gif function is a helper utility designed to create an animated GIF from a series of image files.
    The images are read from a specified directory and filtered based on a given chart name. The resulting GIF is saved
    to a specified output file.

    Parameters:
    - filename (str): The path where the output GIF will be saved. Default is ./animation.
    - database (str): The directory containing the image files. This should be a valid directory path.
    - dataName (str): A substring to filter the image files in the origin directory. Only files containing this
      substring in their names will be included in the GIF.
    - fps (int): Frames per second for the GIF. This determines the speed of the animation. Default is 20.
    - loop (int): Number of times the GIF will loop. Default is 0, which means the GIF will loop indefinitely.
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
    print(f"(save_gif): Number of files found: {len(filesForAnimation)}. Creating animation...")

    if len(filesForAnimation) > 1:
        # Open and compile the images into a GIF
        imgs = [Image.open(database+file) for file in filesForAnimation]
        imgs[0].save(fp=filename, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000/fps), loop=loop)

        # Print a confirmation message
        print(f"(save_gif): Animation file \"{filename}\" was created with {fps} fps")
    else:
        print(f"(save_gif): WARNING: Less than 2 files found for '{dataName}', no GIF created!")

def save_mp4(filename: str = "./animation",
             database: str = None,
             dataName: str = None,
             fps: int = 20,
             loop: int = 0):
    """
    Description:
    The save_gif function is a helper utility designed to create an animated GIF from a series of image files.
    The images are read from a specified directory and filtered based on a given chart name. The resulting GIF is saved
    to a specified output file.

    Parameters:
    - filename (str): The path where the output GIF will be saved. Default is ./animation.
    - database (str): The directory containing the image files. This should be a valid directory path.
    - dataName (str): A substring to filter the image files in the origin directory. Only files containing this
      substring in their names will be included in the GIF.
    - fps (int): Frames per second for the GIF. This determines the speed of the animation. Default is 20.
    - loop (int): Number of times the GIF will loop. Default is 0, which means the GIF will loop indefinitely.
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
print(f"(INFO) Trying to simulate {n_steps} ({n_start} to {n_stop_target}) steps, representing {t_stop_target:.3f} seconds [PU]!")

### SIMULATOR SETUP
print("STATUS: Simulator setup started...")
# ceate objects, link and assemble

# STENCIL
if dim == 2:
    if stencil == "D2Q9":
        stencil_class = lt.D2Q9
    else:
        print("WARNING: wrong stencil choice for 2D simulation, D2Q9 is used")
        stencil_class = lt.D2Q9
elif dim == 3:
    if stencil == "D3Q15":
        stencil_class = lt.D3Q15
    elif stencil == "D3Q19":
        stencil_class = lt.D3Q19
    elif stencil == "D3Q27":
        stencil_class = lt.D3Q27
    else:
        print("WARNING: wrong stencil choice for 3D simulation, D3Q27 is used")
        stencil_class = lt.D3Q27
else:
    print("WARNING: wrong dimension choise. Using 2D simulation and D2Q9")
    stencil_class = lt.D2Q9
    dim = 2

if float_dtype == "float32" or float_dtype == "single":
    float_dtype = torch.float32
elif float_dtype == "double" or float_dtype == "float64":
    float_dtype = torch.float64
elif float_dtype == "half" or float_dtype == "float16":
    float_dtype = torch.float16

# LATTICE
lattice = lt.Lattice(stencil_class, device=torch.device(default_device), dtype=float_dtype)
if eqlm:  # use EQLM with 20% less memory usage and 2% less performance
    print("(INFO) Using Equilibrium_LessMemory (saving ~20% VRAM on GPU, but ~2% slower)")
    lattice.equilibrium = lt.QuadraticEquilibrium_LessMemory(lattice)

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


# TODO: auslagern von "CreateSBD...", damit ich das nicht an zwei verschiedenen Stellen anpassen muss...
#   an sich wäre es gut diesen code "einfach" zu haben und dann von hier nur extern aufzurufen, damit ich nicht copy-paste und versions-fehler bekomme

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
# INFO: the house_foundation is slightly lowered into the ground, so that solid_combination etc. works well. Ground ist still precise by itself!
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
geometry_hash = hashlib.md5(f"{combine_solids}{args['no_house']}{domain_constraints}{shape}{house_position}{ground_height_pu}{house_length_pu}{house_length_lu}{eg_height_pu}{house_width_pu}{roof_height_pu}{overhang_pu}".encode()).hexdigest()
house_bc_name = "house_BC_"+ str(geometry_hash)
ground_bc_name = "ground_BC_" + str(geometry_hash)

# SAVE geometry input to file:
output_file = open(outdir+"/geometry_pu.txt", "a")
output_file.write(f"\nGEOMETRY of house and ground, after inference of missing lengths (see log):\n")
output_file.write("\n{:30s} = {}".format(str("combine_solids"),str(combine_solids)))
output_file.write("\n{:30s} = {}".format(str("no_house"),args['no_house']))
output_file.write("\n{:30s} = {}".format(str("dx_pu [m]"),f"{(house_length_pu/house_length_lu):.4f}"))
output_file.write("\n{:30s} = {}".format(str("domain_constraints PU"),str(domain_constraints)))
output_file.write("\n{:30s} = {}".format(str("domain shape LU"),str(shape)))
output_file.write("\n{:30s} = {}".format(str("house_position_PU (on XZ ground plane)"),str(house_position)))
output_file.write("\n{:30s} = {}".format(str("ground_height PU"),f"{ground_height_pu:.4f}"))
output_file.write("\n{:30s} = {}".format(str("house_length LU"),str(house_length_lu)))
output_file.write("\n{:30s} = {}".format(str("house_length PU"),f"{house_length_pu:.4f}"))
output_file.write("\n{:30s} = {}".format(str("house width PU"),f"{house_width_pu:.4f}"))
output_file.write("\n{:30s} = {}".format(str("eg height PU"),f"{eg_height_pu:.4f}"))
output_file.write("\n{:30s} = {}".format(str("roof height PU"),f"{roof_height_pu:.4f}"))
output_file.write("\n{:30s} = {}".format(str("roof angle"),f"{roof_angle:.4f}"))
output_file.write("\n{:30s} = {}".format(str("overhangs PU"),f"{overhang_pu:.4f}"))
output_file.write(f"\n")
output_file.write(f"\ngeometry hash: {geometry_hash}")
output_file.write(f"\n")
output_file.write(f"\nHOUSE & GROUND corner coordinates (2D):")
output_file.write(f"\nhouse polygon PU: \n{np.array(house_polygon)}")
output_file.write(f"\nmin/max z house PU: {(minz_house, maxz_house)}")
output_file.write(f"\n")
output_file.write(f"\nground polygon PU: \n{np.array(ground_polygon)}")
output_file.write(f"\n")
output_file.write(f"\nhouse polygon LU: \n{res * np.array(house_polygon)}")
output_file.write(f"\nmin/max z house LU: {(res*minz_house, res*maxz_house)}")
output_file.write(f"\n")
output_file.write(f"\nground polygon LU: \n{res * np.array(ground_polygon)}")
output_file.write(f"\n")
# res = LU/m
output_file.close()

## CALCULATE SOLID BOUNDARY DATA
print("Calculating 3D TopoDS_Shapes...")

house_prism_shape = build_house_max(house_polygon, minz=minz_house, maxz=maxz_house)  #TopoDS_Shape als Rückgabe
ground_prism_shape = build_house_max(ground_polygon, minz=zmin-0.1*domain_width_pu, maxz=zmax+0.1*domain_width_pu)
if combine_solids:
    print("(INFO) combine_solids==True -> Combining Shapes of house and ground...")
    house_prism_shape = TopoDS_Shape(BRepAlgoAPI_Fuse(house_prism_shape, ground_prism_shape).Shape())

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
else:
    print("(!) (INFO) no_house == True, no house geometry will be included in the simulation")

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

# inspect solid_boundary/IBB-data
if args["plot_intersection_info"]:
    if not args["no_house"]:
        print("(INFO) plotting intersection info...")
        plot_intersection_info(house_solid_boundary_data, makeGrid(domain_constraints, shape), lattice, house_solid_boundary_data.solid_mask, outdir, name=house_bc_name, show=not cluster)
        if not combine_solids:
            plot_intersection_info(ground_solid_boundary_data, makeGrid(domain_constraints, shape), lattice, ground_solid_boundary_data.solid_mask, outdir, name=ground_bc_name, show=not cluster)



## FLOW Class
print("Initializing flow class...")
flow = HouseFlow3D(shape, re, ma, lattice, domain_constraints,
                   char_length_lu=house_length_lu,
                   char_length_pu=house_length_pu,
                   char_velocity_pu=char_velocity_pu,
                   char_density_pu=char_density_pu,
                   u_init=u_init,
                   reference_height_pu=reference_height_pu, ground_height_pu=ground_height_pu,
                   inlet_bc=inlet_bc, outlet_bc=outlet_bc,
                   ground_bc=ground_bc if not combine_solids else None,
                   house_bc=house_bc, top_bc=top_bc,
                   inlet_ramp_steps=args["inlet_ramp_steps"],
                   house_solid_boundary_data=house_solid_boundary_data,
                   ground_solid_boundary_data=ground_solid_boundary_data,  # will be None for combine_solids == True
                   K_Factor=10,  # K_factor for SEI boundary inlet
                   L=3,  # L for SEI
                   N=34,  # N number of random voctices for SEI
                   wsp_shift_up_pu=args["wsp_shift_up_lu"] * 1 / res,  #how many PU to shift the u_inlet profile upwards, without altering the ground_height
                   wsp_y0=args["wsp_y0_lu"] * 1 / res,
                   wsp_alpha=args["wsp_alpha"],
                   )

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
output_file.write('\n{:30s} {:30s}'.format("rho_char_LU", str(flow.units.characteristic_density_lu)))
output_file.write('\n')
output_file.write('\n{:30s} {:30s}'.format("l_char_PU", str(flow.units.characteristic_length_pu)))
output_file.write('\n{:30s} {:30s}'.format("u_char_PU", str(flow.units.characteristic_velocity_pu)))
output_file.write('\n{:30s} {:30s}'.format("viscosity_PU", str(flow.units.viscosity_pu)))
output_file.write('\n{:30s} {:30s}'.format("p_char_PU", str(flow.units.characteristic_pressure_pu)))
output_file.write('\n{:30s} {:30s}'.format("rho_char_PU", str(flow.units.characteristic_density_pu)))
output_file.write('\n')
output_file.write('\n{:30s} {:30s}'.format("grid reynolds number Re_g", str(flow.units.characteristic_velocity_lu/(lattice.stencil.cs**2 * (flow.units.relaxation_parameter_lu - 0.5)))))
output_file.write('\n{:30s} {:30s}'.format("flow through time PU", str(domain_length_pu/char_velocity_pu)))
output_file.write('\n{:30s} {:30s}'.format("flow through time LU", str(flow.grid[0].shape[0]/flow.units.characteristic_velocity_lu)))
output_file.write('\n')
output_file.close()


# COLLISION
print("Initializing collision operator...")
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

print("Initializing streaming object...")
streaming = lt.StandardStreaming(lattice)
print("Initializing simulation object...")
simulation = lt.Simulation(flow, lattice, collision_obj, streaming)
if t_sim_max > 0:
    simulation.t_max = t_sim_max

# OPTIONAL
#simulation.initialize_f_neq()

# OPTIONAL initialization process with reynolds *1/100
initialize_low_re = False
if initialize_low_re:
    # adjust re and tau for initialization of simulation with lower Re -> influences more dissipative tau
    re_original = re
    re_init = 1/100*re
    flow.units.reynolds_number = re_init
    collision_obj.tau = flow.units.relaxation_parameter_lu

    #TODO: export checkpoint after initialization that can be imported...
    #TODO: "from_init_cpt" Option mit Pfad..., dann wird die Initialisierung übersprungen

    # reinitialize simulation with correct Re
    simulation.i = n_start
    flow.units.reynolds_number = re_original
    collision_obj.tau = flow.units.relaxation_parameter_lu

## PLOT VELOCITY PROFILE over central slice in XY-plane at z=int(Z/2)
import matplotlib
matplotlib.rcParams.update({'figure.figsize': (5,4)})
fig, ax = plt.subplots(constrained_layout=True)
ux_profile_lu = flow.units.convert_velocity_to_lu(flow.wind_speed_profile_power_law(np.where(flow.solid_mask, 0, flow.grid[1]),
                                                      y_ref=flow.reference_height_pu,
                                                      # REFERENCE height (roof or eg_height)
                                                      y_0=flow.wsp_y0,
                                                      u_ref=flow.units.characteristic_velocity_pu,
                                                      # characteristic velocity at reference height (EG or ROOF)
                                                      alpha=flow.wsp_alpha)[0, np.newaxis,...][0, :, int(shape[2] / 2)])
                                          #lattice.convert_to_numpy(lattice.u(simulation.f)[0, 0, :, int(shape[2] / 2)])  # (!) lattice.u gibt LU, flow.initial_solution gibt PU
y_values = np.arange(len(ux_profile_lu))
ux_profile_table = np.stack([y_values, ux_profile_lu])
np.savetxt(outdir + f"/velocity_profile_ux_inlet.txt", ux_profile_table, header="y_value (LU) |  ux (LU)")
ax.plot(ux_profile_lu, y_values, marker ="x", linestyle =":")
ax.set_xlabel("ux [LU]")
ax.set_ylabel("y [LU]")
secax = ax.secondary_yaxis('right', functions=(flow.units.convert_length_to_pu, flow.units.convert_length_to_lu))
secax.set_ylabel("y [PU]")
secax2 = ax.secondary_xaxis('top', functions=(flow.units.convert_velocity_to_pu, flow.units.convert_velocity_to_lu))
secax2.set_xlabel("ux [PU]")
plt.grid()
#fig.suptitle(str(timestamp) + "\n" + name + "\n" + "velocity inlet profile")
plt.savefig(outdir+"/velocity_profile_ux_inlet.png")
if not cluster:
    plt.show()

## determine ux-velocity profile gradients...
fig, ax = plt.subplots(constrained_layout=True)
ux_profile_lu_deltas = ux_profile_lu[1:] - ux_profile_lu[:-1]
y_values = np.arange(len(ux_profile_lu_deltas)) + 0.5
ux_profie_deltas_table = np.stack([y_values, ux_profile_lu_deltas])
np.savetxt(outdir + f"/velocity_profile_ux_deltas_inlet.txt", ux_profie_deltas_table, header="y_value (LU)  |  dux/dy (LU)")
ax.plot(ux_profile_lu_deltas, y_values, marker ="x", linestyle =":")
ax.set_xlabel("dux/dy [LU]")
ax.set_ylabel("y [LU]")
# Begrenze die Anzahl der Ticks auf der unteren x-Achse
import matplotlib.ticker as ticker
ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
secax = ax.secondary_yaxis('right', functions=(flow.units.convert_length_to_pu, flow.units.convert_length_to_lu))
secax.set_ylabel("y [PU]")
secax2 = ax.secondary_xaxis('top', functions=(flow.units.convert_velocity_to_pu, flow.units.convert_velocity_to_lu))
secax2.set_xlabel("dux/dy [PU]")
plt.grid()
#fig.suptitle(str(timestamp) + "\n" + name + "\n" + f"velocity inlet profile deltas (grad)\nUXmaxLU = {max(ux_profile_lu_deltas):.5f}, UXmaxPU = {flow.units.convert_velocity_to_pu(max(ux_profile_lu_deltas)):.5f}")
plt.savefig(outdir+"/velocity_profile_ux_deltas_inlet.png")
if not cluster:
    plt.show()

print(f"(!) maximum velocity gradient on inlet is dux/dy = {max(ux_profile_lu_deltas)} [LU], {flow.units.convert_velocity_to_pu(max(ux_profile_lu_deltas))} [PU]")


print("\nmaximum total (CPU) RAM usage ('MaxRSS') (including optional PNG and GIF post-processing [MB]: " + str(round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, 2)) + " MB")

## END OF SCRIPT
print(f"\n♬ THE END ♬")
sys.stdout = old_stdout