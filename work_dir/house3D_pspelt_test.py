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
parser.add_argument("--float_dtype", default="float32", choices=["float32", "float64", "single", "double", "half"], help="data type for floating point calculations in torch")
parser.add_argument("--t_sim_max", default=(72*60*60), type=float, help="max. walltime to simulate, default is 72 h for cluster use. sim stops at 0.99*t_max_sim")  # andere max.Zeit? wie lange braucht das "drum rum"? kann cih simulation auch die scho vergangene Zeit übergeben? dann kann ich mit nem größeren Wert rechnen und sim ist variabel darin, wie viel Zeit es noch hat

parser.add_argument("--cluster", action='store_true', help="if you don't want pngs etc. to open, please use this clsuter-flag")
parser.add_argument("--outdir", default=os.getcwd(), type=str, help="directory to save output files to; vtk-files will be saved in seperate dir, if outputdir_vtk is specified")
parser.add_argument("--outdir_vtk", default=None, type=str, help="")
parser.add_argument("--vtk", action='store_true', help="toggle vtk-output to outdor_vtk, if set True (1)")
parser.add_argument("--vtk_fps", default=10, help="frames per second_PU for VTK output; overwritten if vtk_interval is specified")
parser.add_argument("--vtk_interval", default=0, type=int, help="how many steps between vtk-output-files; overwrites vtk_fps")
parser.add_argument("--vtk_t_interval", default=0, type=float, help="how many seconds between vtk-output-files; overwrites vtk_fps and vtk_interval; for low frequency output")
parser.add_argument("--vtk_step_start", default=0, type=float, help="at which step to start vtk export")
parser.add_argument("--vtk_step_end", default=0, type=float, help="at which step to end vtk export")
parser.add_argument("--vtk_t_start", default=0, type=float, help="at which time (PU) to start vtk export")
parser.add_argument("--vtk_t_end", default=0, type=float, help="at which time (PU) to stop vtk export")

parser.add_argument("--nan_reporter", action='store_true', help="stop simulation if NaN is detected in f field")
parser.add_argument("--nan_reporter_interval", default=100, type=int, help="interval in which the NaN reporter checks f for NaN")
parser.add_argument("--high_ma_reporter", action='store_true', help="stop simulation if Ma > 0.3 is detected in u field")
parser.add_argument("--watchdog", action='store_true', help="report progress, ETA and warn, if Sim is estimated to run longer than t_max (~72 h)")
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

# boundary algorithms
parser.add_argument("--inlet_bc", default="eqin", help="inlet boundary condition: EQin, NEX, SEI")
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
parser.add_argument("--animations_fps_gif", default=0, type=int, help="number of frames per second PU for 2D animations (GIFs). Actual fps of the resulting GIF. (Not the fps at which frames are taken from simulation!")
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
name, default_device, float_dtype, t_sim_max, cluster, outdir, outdir_vtk, vtk, vtk_fps, vtk_interval, vtk_step_start, \
    nan_reporter, from_cpt, sim_i, write_cpt, re, ma, viscosity_pu, char_density_pu, u_init, n_steps, t_target, \
    step_start, collision, dim, stencil, eqlm, house_length_lu, house_length_pu, house_width_pu, roof_angle, \
    eg_height_pu, roof_height_pu, overhang_pu, domain_length_pu, domain_width_pu, domain_height_pu, inlet_bc, outlet_bc, \
    ground_bc, house_bc, top_bc, combine_solids, verbose = \
    [args[_] for _ in ["name", "default_device", "float_dtype", "t_sim_max", "cluster", "outdir", "outdir_vtk",
                       "vtk", "vtk_fps", "vtk_interval", "vtk_step_start", "nan_reporter", "from_cpt", "sim_i",
                       "write_cpt", "re", "ma", "viscosity_pu", "char_density_pu", "u_init", "n_steps", "t_target",
                       "step_start", "collision", "dim", "stencil", "eqlm", "house_length_lu", "house_length_pu",
                       "house_width_pu", "roof_angle", "eg_height_pu", "roof_height_pu", "overhang_pu", "domain_length_pu",
                       "domain_width_pu", "domain_height_pu", "inlet_bc", "outlet_bc", "ground_bc", "house_bc",
                       "top_bc", "combine_solids", "verbose"]]

# CREATE timestamp, sim-ID, outdir and outdir_vtk
timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
sim_id = str(timestamp) + "-" + name
os.makedirs(outdir+"/"+sim_id)
if outdir_vtk is None:
    outdir_vtk = outdir
outdir = outdir+"/"+sim_id  # adding individal sim-ID to outdir path to get individual DIR per simulation
outdir_vtk = outdir_vtk+"/"+sim_id
if (vtk or args["save_animations"]) and not os.path.exists(outdir_vtk):
    os.makedirs(outdir_vtk)
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
            u_LU = lattice.u(f)
            rho_LU = lattice.rho(f)

            u_PU = flow.units.convert_velocity_to_pu(u_LU)
            p_PU = flow.units.convert_density_lu_to_pressure_pu(rho_LU)

            u = lattice.convert_to_numpy(u_PU)
            p = lattice.convert_to_numpy(p_PU)
            u_magnitude = np.linalg.norm(u, axis=0)

            self.show2d_slice_reporter(u_magnitude, f"u_mag(t = {t:.3f} s, step = {simulation.i}) noLIM",f"nolim_u_mag_i{simulation.i:08}_t{int(t)}", cmap=self.cmap)
            self.show2d_slice_reporter(u_magnitude, f"u_mag(t = {t:.3f} s, step = {simulation.i}) LIM99",f"lim99_u_mag_i{simulation.i:08}_t{int(t)}", vlim=(np.percentile(u_magnitude.flatten(), 1), np.percentile(u_magnitude.flatten(), 99)), cmap=self.cmap)
            self.show2d_slice_reporter(u_magnitude, f"u_mag(t = {t:.3f} s, step = {simulation.i}) LIM2uchar", f"lim2uchar_u_mag_i{simulation.i:08}_t{int(t)}", vlim=self.u_lim, cmap=self.cmap)

            self.show2d_slice_reporter(p[0], f"p (t = {t:.3f} s, step = {simulation.i}) noLIM", f"nolim_p_i{simulation.i:08}_t{int(t)}", cmap=self.cmap)
            self.show2d_slice_reporter(p[0], f"p (t = {t:.3f} s, step = {simulation.i}) LIM99",f"lim99_p_i{simulation.i:08}_t{int(t)}", vlim=(np.percentile(p[0].flatten(), 1), np.percentile(p[0].flatten(), 99)), cmap=self.cmap)
            self.show2d_slice_reporter(p[0], f"p (t = {t:.3f} s, step = {simulation.i}) LIMfix",f"limfix_p_i{simulation.i:08}_t{int(t)}", vlim=self.p_lim, cmap=self.cmap)







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

# DOMAIN constraints in PU
print("Defining domain constraints...")
xmin, ymin, zmin = 0, 0, 0 if dim == 3 else None
xmax, ymax, zmax = domain_length_pu, domain_height_pu, domain_width_pu if dim == 3 else None
minz_house, maxz_house = (domain_width_pu/2.-house_width_pu/2., domain_width_pu/2.+house_width_pu/2.) if dim == 3 else (-1, 1)
ground_height_pu = 0.5 * house_length_pu/house_length_lu  # height of ZERO-height or ground level in PU at 0.5 LU

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

# TODO: reformat to output_file.write('{:30s} {:30s}\n'.format(str(key), str(args[key])))
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
output_file.write(f"\nmin/max z house PU: {(res*minz_house, res*maxz_house)}")
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
                   u_init=u_init,
                   reference_height_pu=reference_height_pu, ground_height_pu=ground_height_pu,
                   inlet_bc=inlet_bc, outlet_bc=outlet_bc,
                   ground_bc=ground_bc if not combine_solids else None,
                   house_bc=house_bc, top_bc=top_bc,
                   house_solid_boundary_data=house_solid_boundary_data,
                   ground_solid_boundary_data=ground_solid_boundary_data,  # will be None for combine_solids == True
                   K_Factor=10,  # K_factor for SEI boundary inlet
                   L=3,  # L for SEI
                   N=34,  # N number of random voctices for SEI
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
output_file.write('\n')
output_file.write('\n{:30s} {:30s}'.format("l_char_PU", str(flow.units.characteristic_length_pu)))
output_file.write('\n{:30s} {:30s}'.format("u_char_PU", str(flow.units.characteristic_velocity_pu)))
output_file.write('\n{:30s} {:30s}'.format("viscosity_PU", str(flow.units.viscosity_pu)))
output_file.write('\n{:30s} {:30s}'.format("p_char_PU", str(flow.units.characteristic_pressure_pu)))
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


## CHECK INITIALISATION AND 2D DOMAIN
print(f"Initializing Show2D instances for 2d plots...")
if args["save_animations"]:
    observable_2D_plots_path = outdir_vtk + "/observable_2D_plots"
else:
    observable_2D_plots_path = outdir + "/observable_2D_plots"
if args["plot_sbd_2d"]:
    boundary_masks_2D_plots_path = outdir + "/boundary_data_2D_plots"

if not os.path.isdir(observable_2D_plots_path):
    os.makedirs(observable_2D_plots_path)
if args["plot_sbd_2d"] and not os.path.isdir(boundary_masks_2D_plots_path):
    os.makedirs(boundary_masks_2D_plots_path)
show2d_observables = Show2D(lattice, flow.solid_mask, domain_constraints, outdir=observable_2D_plots_path, show=not cluster, figsize=(4,4), dpi=dpi)
if args["plot_sbd_2d"]:
    show2d_boundaries = Show2D(lattice, flow.solid_mask, domain_constraints, outdir=boundary_masks_2D_plots_path, show=not cluster, figsize=(8,8), dpi=dpi/2)

# plot initial u_x velocity field as 2D slice
show2d_observables(lattice.convert_to_numpy(flow.units.convert_velocity_to_pu(lattice.u(simulation.f))[0]), "u_x_INIT(t=0)", "u_x_t0")

# PRINT ALL boundary.mask(s) to check positioning...
if args["plot_sbd_2d"]:
    print(f"Analyzing and plotting masks and f_indices to dir = '{boundary_masks_2D_plots_path}' ...")
    baguette = 0
    for boundary in simulation.boundaries:
        print(f"Analyzing and plotting masks and f_indices for simulation.boundary[{baguette}]...")
        if hasattr(boundary, "mask"):
            show2d_boundaries(lattice.convert_to_numpy(boundary.mask) if not isinstance(boundary.mask, np.ndarray) else boundary.mask, title="boundary.mask of boundary[{baguette}]:\n"+str(boundary), name="boundary_mask_"+str(baguette))
        # FWBB
        if hasattr(boundary, "f_index_fwbb"):
            temp_q_mask = np.zeros_like(lattice.convert_to_numpy(simulation.f), dtype=bool)
            temp_mask = np.zeros_like(boundary.mask)
            if boundary.f_index_fwbb.shape[0] > 0:  # vereine alle f_indices in einer node-Maske
                idx_numpy = lattice.convert_to_numpy(boundary.f_index_fwbb)
                temp_mask[idx_numpy[:, 1], idx_numpy[:, 2], idx_numpy[:, 3] if len(boundary.mask.shape) == 3 else None] = True
                temp_q_mask[idx_numpy[:, 0], idx_numpy[:, 1], idx_numpy[:, 2], idx_numpy[:, 3] if len(boundary.mask.shape) == 3 else None] = True
            show2d_boundaries(temp_mask, title=f"f_index_fwbb of boundary[{baguette}]:\n"+str(boundary), name="f_index_fwbb_"+str(baguette))
        # HWBB
        if hasattr(boundary, "f_index"):
            temp_q_mask = np.zeros_like(lattice.convert_to_numpy(simulation.f), dtype=bool)
            temp_mask = np.zeros_like(boundary.mask)
            if boundary.f_index.shape[0] > 0:  # vereine alle f_indices in einer node-Maske
                idx_numpy = lattice.convert_to_numpy(boundary.f_index)
                temp_mask[idx_numpy[:, 1], idx_numpy[:, 2], idx_numpy[:, 3] if len(boundary.mask.shape) == 3 else None] = True
                temp_q_mask[idx_numpy[:, 0], idx_numpy[:, 1], idx_numpy[:, 2], idx_numpy[:, 3] if len(boundary.mask.shape) == 3 else None] = True
            show2d_boundaries(temp_mask, title=f"f_index of boundary[{baguette}]:\n" + str(boundary), name="f_index_" + str(baguette))
        # IBB
        if hasattr(boundary, "f_index_lt"):
            temp_q_mask = np.zeros_like(lattice.convert_to_numpy(simulation.f), dtype=bool)
            temp_mask = np.zeros_like(boundary.mask)
            if boundary.f_index_lt.shape[0] > 0:  # vereine alle f_indices in einer node-Maske
                idx_numpy = lattice.convert_to_numpy(boundary.f_index_lt)
                temp_mask[idx_numpy[:, 1], idx_numpy[:, 2], idx_numpy[:, 3] if len(boundary.mask.shape) == 3 else None] = True
                temp_q_mask[idx_numpy[:, 0], idx_numpy[:, 1], idx_numpy[:, 2], idx_numpy[:, 3] if len(boundary.mask.shape) == 3 else None] = True
            show2d_boundaries(temp_mask, title=f"f_index_lt of boundary[{baguette}]:\n" + str(boundary), name="f_index_lt_" + str(baguette))
        if hasattr(boundary, "f_index_gt"):
            temp_q_mask = np.zeros_like(lattice.convert_to_numpy(simulation.f), dtype=bool)
            temp_mask = np.zeros_like(boundary.mask)
            if boundary.f_index_gt.shape[0] > 0:  # vereine alle f_indices in einer node-Maske
                idx_numpy = lattice.convert_to_numpy(boundary.f_index_gt)
                temp_mask[idx_numpy[:, 1], idx_numpy[:, 2], idx_numpy[:, 3] if len(boundary.mask.shape) == 3 else None] = True
                temp_q_mask[idx_numpy[:, 0], idx_numpy[:, 1], idx_numpy[:, 2], idx_numpy[:, 3] if len(boundary.mask.shape) == 3 else None] = True
            show2d_boundaries(temp_mask, title=f"f_index_gt of boundary[{baguette}]:\n" + str(boundary), name="f_index_gt_" + str(baguette))
        if hasattr(boundary, "f_index_lt") and hasattr(boundary, "f_index_gt"):
            temp_q_mask = np.zeros_like(lattice.convert_to_numpy(simulation.f), dtype=bool)
            temp_mask = np.zeros_like(boundary.mask)
            if boundary.f_index_lt.shape[0] > 0 and boundary.f_index_gt.shape[0] > 0:  # vereine alle f_indices in einer node-Maske
                idx_numpy = np.concatenate([lattice.convert_to_numpy(boundary.f_index_lt), lattice.convert_to_numpy(boundary.f_index_gt)], axis=0)
                temp_mask[idx_numpy[:, 1], idx_numpy[:, 2], idx_numpy[:, 3] if len(boundary.mask.shape) == 3 else None] = True
                temp_q_mask[idx_numpy[:, 0], idx_numpy[:, 1], idx_numpy[:, 2], idx_numpy[:, 3] if len(boundary.mask.shape) == 3 else None] = True
            show2d_boundaries(temp_mask, title=f"f_index_ltgt of boundary[{baguette}] in XY:\n" + str(boundary), name="f_index_ltgt_" + str(baguette))
            show2d_boundaries(temp_mask, title=f"f_index_ltgt of boundary[{baguette}] in YZ:\n" + str(boundary), name="f_index_ltgt_" + str(baguette) + "YZ", position=int(flow.units.convert_length_to_lu(house_position[0])), normal_dir=0)

        # TODO: (OPT) Schleife über alle q in temp_q_mask und dann alle in ein DIR ausgeben, sodass man sieht, wo jeweils hingezeigt wird.
        #  Das könnte dann noch mit d_lt und d_gt auf den d-Wert gesetzt werden, sodass man eine heatmap der ds hat!
        baguette += 1



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
if vtk:

    print(args["vtk_t_start"])
    print(args["vtk_t_end"])
    print(args["vtk_step_start"])
    print(args["vtk_step_end"])
    print(args["vtk_interval"])
    print(args["vtk_t_interval"])

    if args["vtk_t_start"] > 0:
        print("(vtk) overwriting vtk_step_start with {}, because vtk_t_start = {}")
        vtk_i_start = int(round(flow.units.convert_time_to_lu(args["vtk_t_start"])))
    else:
        vtk_i_start = int(args["vtk_step_start"])

    if args["vtk_t_end"] > 0:
        print("(vtk) overwriting vtk_step_end with {}, because vtk_t_end = {}")
        vtk_i_end = int(flow.units.convert_time_to_lu(args["vtk_t_end"]))
    elif args["vtk_step_end"] > 0:
        vtk_i_end = args["vtk_step_end"]
    else:
        vtk_i_end = n_steps

    if args["vtk_t_interval"] > 0:
        vtk_interval = int(flow.units.convert_time_to_lu(args["vtk_t_interval"]))
    elif args["vtk_interval"] > 0:
        vtk_interval = args["vtk_interval"]
    else:
        vtk_interval = int(flow.units.convert_time_to_lu(1/vtk_fps))

    if vtk_interval < 1:
        vtk_interval = 1

    print(vtk_i_start)
    print(vtk_i_end)
    print(vtk_interval)
    print(vtk_fps)

    print(f"(INFO) Appending vtk reporter with vtk_interval = {int(flow.units.convert_time_to_lu(1/vtk_fps)) if vtk_interval == 0 else int(vtk_interval)}, vtk_start = <>, vtk_end = <> and vtk_dir: {outdir_vtk}/vtk/out")
    print(f"(INFO) This will create approx. {(vtk_i_end-vtk_i_start)/vtk_interval+1} .vti or .vtk files!")
    vtk_reporter = lt.VTKReporter(lattice, flow,
                                  interval=int(vtk_interval),
                                  filename_base=outdir_vtk+"/vtk/out", imin=vtk_i_start, imax=vtk_i_end)
    simulation.reporters.append(vtk_reporter)

    # export solid_mask
    # mask_dict = dict()
    # mask_dict["mask"] = flow.solid_mask.astype(int) if len(flow.shape) == 3 else flow.solid_mask[..., None].astype(int)  # extension to pseudo-3D is needed for vtk-export to work
    # imageToVTK(
    #     path=outdir_vtk + "/solid_mask_point",
    #     pointData=mask_dict
    # )
    # imageToVTK(
    #     path=outdir_vtk + "/solid_mask_cell",
    #     cellData=mask_dict
    # )

    # NEW
    vtk_reporter.output_mask(flow.solid_mask, outdir_vtk, "solid_mask")

    if not combine_solids:
        # export house_mask
        # mask_dict["mask"] = flow.house_mask.astype(int) if len(flow.shape) == 3 else flow.house_mask[..., None].astype(int)  # extension to pseudo-3D is needed for vtk-export to work
        # imageToVTK(
        #     path=outdir_vtk + "/house_mask_point",
        #     pointData=mask_dict
        # )
        # imageToVTK(
        #     path=outdir_vtk + "/house_mask_cell",
        #     cellData=mask_dict
        # )

        #NEW
        vtk_reporter.output_mask(flow.house_mask, outdir_vtk, "house_mask")

        # export ground_mask
        # mask_dict["mask"] = flow.ground_mask.astype(int) if len(flow.shape) == 3 else flow.ground_mask[..., None].astype(
        #     int)  # extension to pseudo-3D is needed for vtk-export to work
        # imageToVTK(
        #     path=outdir_vtk + "/ground_mask_point",
        #     pointData=mask_dict
        # )
        # imageToVTK(
        #     path=outdir_vtk + "/ground_mask_cell",
        #     cellData=mask_dict
        # )

        # NEW
        vtk_reporter.output_mask(flow.ground_mask, outdir_vtk, "ground_mask")

# PROGRESS REPORTER
# progress_reporter = lt.ProgressReporter(flow, n_stop_target)
# simulation.reporters.append(progress_reporter)

# WATCHDOG-REPORTER
if args["watchdog"]:
    watchdog_reporter = lt.Watchdog(lattice, flow, simulation, interval=int(n_steps/100), i_start=n_start, i_target=n_stop_target, t_max=simulation.t_max, filebase=outdir+"/watchdog", show=not cluster)
    simulation.reporters.append(watchdog_reporter)

# NAN REPORTER
if args["nan_reporter"]:
    nan_reporter = lt.NaNReporter(flow, lattice, n_stop_target, t_stop_target, interval=args["nan_reporter_interval"], simulation=simulation, vtk_dir=outdir_vtk, vtk=True, outdir=outdir)  # omitting outdir leads to no extra file with coordinates being created. With a resolution of >100.000.000 Gridpoints, torch gets confused otherwise...
    simulation.reporters.append(nan_reporter)

if args["high_ma_reporter"]:
    high_ma_reporter_path = outdir+"/HighMaReporter"
    # if not os.path.exists(high_ma_reporter_path):
    #     os.makedirs(high_ma_reporter_path)
    high_ma_reporter = lt.HighMaReporter(flow, lattice, n_stop_target, t_stop_target, interval=args["nan_reporter_interval"], simulation=simulation, outdir=high_ma_reporter_path, vtk_dir=outdir_vtk, stop_simulation=False)  # stop_simulation overwrites vtk output of HighMaReporter with False
    simulation.reporters.append(high_ma_reporter)

# slice2dReporter for u_mag and p fields:
if args["save_animations"]:
    if args["animations_number_of_frames"] > 0:  # number of 2D frames to take for animations
        interval = int(n_steps/args["animations_number_of_frames"])
    elif args["animations_fps_pu"] > 0:  # if no number of frames given, calculate it from fps
        number_of_frames = t_target * args["animations_fps_pu"]
        interval = int(n_steps/number_of_frames)
    else:  #neither number of frames, nor fps are given, take 1000 frames...
        interval = int(n_steps / 1000)

    slice2dReporter = Slice2dReporter(lattice, simulation, domain_constraints=domain_constraints, interval=interval, start=0, outdir=observable_2D_plots_path, show = False)
    simulation.reporters.append(slice2dReporter)

## WRITE PARAMETERS to file in outdir
# TODO: write parameters to file (ähnlich zu mdout.mdp)
#  siehe auch oben, args wird rausgeschrieben als "input_parameters". Die konkrete ÄNDERUNG aller inputs für die Sim müsste ich dann nochmal hier händisch schreiben


## LOAD CHECKPOINT FILE
# TODO: load checkpoint file and adjust sim.i

# (TEMP) FLOW THROUGH TIME
print(f"flow through time (time a particle takes to travel from input to output unobstructed at u_char): T_ft_PU = {domain_length_pu/char_velocity_pu} s")
print(f"(debug) flow through time (calc. as grid.shape[0]/u_char_lu) T_ft_LU = {flow.grid[0].shape[0]/flow.units.characteristic_velocity_lu} steps")
print(f"(debug) flow through time (calc. as convert_PU_to_LU(T_ft_PU) T_ft_LU = {flow.units.convert_time_to_lu(domain_length_pu/char_velocity_pu)} steps")


### RUN SIMULATION
#n_steps = n_stop_target - n_start
t_start = time()

print(f"\n\n***** SIMULATION STARTED AT {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} *****\n")
mlups = simulation.step(n_steps)

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


# TODO: change output to f-string
print("\n*** HARDWARE UTILIZATION ***\n")
print(f"current GPU VRAM (MB) usage: {torch.cuda.memory_allocated(device=default_device)/1024/1024:.3f}")
print(f"max. GPU VRAM (MB) usage: {torch.cuda.max_memory_allocated(device=default_device)/1024/1024:.3f}")

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


### *** PLOTTING OF FINAL OBSERVABLE FIELDS ***

# # TODO: make plotting parametrisierbar
# ### PLOTTING u, p, vorticity over 2D slice:
# t = flow.units.convert_time_to_pu(simulation.i)
# dx = flow.units.convert_length_to_pu(1.0)
#
# u_LU = lattice.u(simulation.f)
# rho_LU = lattice.rho(simulation.f)
# u_PU = flow.units.convert_velocity_to_pu(u_LU) #[:,:,:,int(u_LU.shape[3]/2)]
# p_PU = flow.units.convert_density_lu_to_pressure_pu(rho_LU)  #[:,:,:,int(rho_LU.shape[3]/2)]
#
# # TODO: vorticity and nicht-periodische Randbedingungen anpassen. denn torch_gradient rechnet so, als ob ALLES free flow wäre...!!!
# plot_vorticity = False
#
# # vorticity:
# if plot_vorticity:
#     grad_u0 = torch_gradient(u_LU[0], dx=dx, order=6)
#     grad_u1 = torch_gradient(u_LU[1], dx=dx, order=6)
#     vorticity = (grad_u0[1] - grad_u1[0]) * (grad_u0[1] - grad_u1[0])
#     if lattice.D == 3:
#         grad_u2 = torch_gradient(u_LU[2], dx=dx, order=6)
#         vorticity += (grad_u2[1] - grad_u1[2]) * (grad_u2[1] - grad_u1[2])\
#                         + ((grad_u0[2] - grad_u2[0]) * (grad_u0[2] - grad_u2[0]))
#     vorticity = vorticity * dx ** lattice.D
#
# # CONVERT TO NUMPY and slice:
# u = lattice.convert_to_numpy(u_PU)  # [:,:,:,int(u_PU.shape[3]/2)]
# p = lattice.convert_to_numpy(p_PU) # [:,:,:,int(rho_LU.shape[3]/2)]
# u_magnitude = np.linalg.norm(u, axis=0)
# if plot_vorticity:
#     vorticity = lattice.convert_to_numpy(vorticity)
#
# print("\nPLOTTING final observable fields...")
# print(f"u.shape = {u.shape}")
# print(f"p.shape = {p.shape}")
# print(f"u_magnitude.shape = {u_magnitude.shape}")
# if plot_vorticity:
#     print(f"vorticity.shape = {vorticity.shape}")
#     print(f"abs(vorticity).shape = {np.abs(vorticity).shape}")
#
# # PROTOTYPE plotting without show2d... >>>
# # fig, axes = plt.subplots(2,1, figsize=(8,8), dpi=300)
# # fig.tight_layout()
# # axes[0].set_title("Pressure")
# # pos0 = axes[0].imshow(p[:,:,:,int(p.shape[3]/2)].transpose(), origin="lower", ) #vmin=p[0,ny-1,0], vmax=p[0].mean(axis=-1).max())
# # fig.colorbar(pos0, ax=axes[0])
# # axes[1].set_title("Velocity magnitude")
# # pos1 = axes[1].imshow(u_magnitude[:,:,int(u_magnitude.shape[2]/2)].transpose(), origin="lower", cmap='inferno',
# #                       vmin=np.percentile(u_magnitude.flatten(), 1),
# #                       vmax=np.percentile(u_magnitude.flatten(), 95)
# #                       )
# # fig.colorbar(pos1, ax=axes[1])
# # plt.show()  # watch out for "show" on cluster...
# # <<<
#
# show2d_observables(u_magnitude, f"u_magnitude(t = {t:.3f} s, step = {simulation.i}) noLIM", f"u_magnitude_t{int(t)}_i{simulation.i:08}_noLIM", cmap='inferno')
# show2d_observables(u_magnitude, f"u_magnitude(t = {t:.3f} s, step = {simulation.i}) LIM99", f"u_magnitude_t{int(t)}_i{simulation.i:08}_LIM99", vlim=(np.percentile(u_magnitude.flatten(),1), np.percentile(u_magnitude.flatten(), 99)))
# show2d_observables(u_magnitude, f"u_magnitude(t = {t:.3f} s, step = {simulation.i}) LIM95", f"u_magnitude_t{int(t)}_i{simulation.i:08}_LIM95", vlim=(np.percentile(u_magnitude.flatten(),1), np.percentile(u_magnitude.flatten(), 95)))
#
# if plot_vorticity:
#     show2d_observables(np.abs(vorticity), f"vorticity_mag(t = {t} s, step = {simulation.i}) noLIM", f"vorticity_mag_t{int(t)}_noLIM")
#     show2d_observables(np.abs(vorticity), f"vorticity_mag(t = {t} s, step = {simulation.i}) LIM99", f"vorticity_mag_t{int(t)}_LIM99", vlim=(np.percentile(np.abs(vorticity).flatten(),1), np.percentile(np.abs(vorticity).flatten(), 99)))
#     show2d_observables(np.abs(vorticity), f"vorticity_mag(t = {t} s, step = {simulation.i}) LIM95", f"vorticity_mag_t{int(t)}_LIM95", vlim=(np.percentile(np.abs(vorticity).flatten(),1), np.percentile(np.abs(vorticity).flatten(), 95)))
#
# show2d_observables(p[0], f"pressure(t = {t:.3f} s, step = {simulation.i}) noLIM", f"pressure_t{int(t)}_i{simulation.i:08}_noLIM")
# show2d_observables(p[0], f"pressure(t = {t:.3f} s, step = {simulation.i}) LIM99", f"pressure_t{int(t)}_i{simulation.i:08}_LIM99", vlim=(np.percentile(p[0].flatten(),1), np.percentile(p[0].flatten(), 99)))
# show2d_observables(p[0], f"pressure(t = {t:.3f} s, step = {simulation.i}) LIM95", f"pressure_t{int(t)}_i{simulation.i:08}_LIM95", vlim=(np.percentile(p[0].flatten(),1), np.percentile(p[0].flatten(), 95)))

### WRITE CHECKPOINT
# TODO: write checkpointfile, if write_cpt=True and save sim.i to it / should be on correct device! or CPU to be save

### POSTPROCESSONG: PROCESS and export observables
# TODO: Observable post-processing, while flow and sim data is still available

### OUTPUT RESULTS
# TODO: output results to human-readable file AND to copy-able file (or csv)

if args["save_animations"]:  # takes images from slice2dReporter and created GIF
    t0 = time()
    print(f"(INFO) creating animations from slice2dRepoter Data...")
    os.makedirs(outdir_vtk+"/animations")

    if args["animations_fps_gif"] > 0:
        fps = int(args["animations_fps_gif"])
    else:
        fps = 3

    save_mp4(outdir_vtk+"/animations/lim99_u_mag", observable_2D_plots_path, "lim99_u_mag", fps=fps)
    save_mp4(outdir_vtk+"/animations/lim2uchar_u_mag", observable_2D_plots_path, "lim2uchar_u_mag", fps=fps)
    save_mp4(outdir_vtk+"/animations/nolim_u_mag", observable_2D_plots_path, "nolim_u_mag", fps=fps)

    save_mp4(outdir_vtk+"/animations/lim99_p", observable_2D_plots_path, "lim99_p", fps=fps)
    save_mp4(outdir_vtk+"/animations/limfix_p", observable_2D_plots_path, "limfix_p", fps=fps)
    save_mp4(outdir_vtk+"/animations/nolim_p", observable_2D_plots_path, "nolim_p", fps=fps)
    t1 = time()
    print(f"Creating animations from slice2dRepoter Data took {floor((t1 - t0) / 60):02d}:{floor((t1- t0) % 60):02d} [mm:ss]")
else:  # plots final u_mag and p fields
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
    if not cluster:
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


# PLOT max. Ma in domain over time...
fig, ax = plt.subplots(constrained_layout=True)
max_u_lu = np.array(max_u_lu_reporter.out)
ax.plot(max_u_lu[:, 1], max_u_lu[:, 2]/lattice.convert_to_numpy(lattice.cs))
ax.set_xlabel("physical time / s")
ax.set_ylabel("maximum Ma")
ax.set_ylim([0,0.3])
secax = ax.secondary_xaxis('top', functions=(flow.units.convert_time_to_lu, flow.units.convert_time_to_pu))
secax.set_xlabel("timesteps (simulation time / LU)")
fig.suptitle(str(timestamp) + "\n" + name)
plt.savefig(outdir+"/max_Ma.png")
if not cluster:
    plt.show()

# PLOT max. u_mag for abs. anaylsis
fig, ax = plt.subplots(constrained_layout=True)
ax.plot(max_u_lu[:, 1], flow.units.convert_velocity_to_pu(max_u_lu[:, 2]))
ax.set_xlabel("physical time / s")
ax.set_ylabel("maximum momentary velocity magnitude (PU)")
y_lim_50 = flow.units.convert_velocity_to_pu(max_u_lu[:int(max_u_lu.shape[0]/1.3), 2].max())  # max u_mag of first 50% of the data (excludes crash, if present, includes settling period)
ax.set_ylim([0, y_lim_50*1.1])  # show 10% more than u_mag_max
secax = ax.secondary_xaxis('top', functions=(flow.units.convert_time_to_lu, flow.units.convert_time_to_pu))
secax.set_xlabel("timesteps (simulation time / LU)")
fig.suptitle(str(timestamp) + "\n" + name)
plt.savefig(outdir+"/max_u_mag.png")
if not cluster:
    plt.show()

# PLOT max/min p for abs. analysis
fig, ax = plt.subplots(constrained_layout=True)
min_max_p_pu = np.array(min_max_p_pu_reporter.out)
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
fig.suptitle(str(timestamp) + "\n" + name)
plt.savefig(outdir+"/min_max_p.png")
if not cluster:
    plt.show()

# TODO: export min/max p, Ma_max and u_max as obs-timeseries. (i,t,Val) - watch storage consumption

print("\nmaximum total (CPU) RAM usage ('MaxRSS') (including optional PNG and GIF post-processing [MB]: " + str(round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, 2)) + " MB")

## END OF SCRIPT
print(f"\n♬ THE END ♬")
sys.stdout = old_stdout