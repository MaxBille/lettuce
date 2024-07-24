### IMPORT

# sys, os etc.
import sys
import os
import psutil
import shutil
from time import time, sleep
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# torch, np,
import numpy as np
import torch

# lettuce
import lettuce as lt
from lettuce import torch_gradient
from lettuce.boundary import InterpolatedBounceBackBoundary_occ, BounceBackBoundary
from lettuce.boundary_mk import EquilibriumExtrapolationOutlet, NonEquilibriumExtrapolationInletU, ZeroGradientOutlet, SyntheticEddyInlet
# flow
from pspelt.obstacleSurface import ObstacleSurface

# pspelt
from pspelt.geometric_building_model import build_house_max
from pspelt.helperFunctions.getIBBdata import getIBBdata
from pspelt.helperFunctions.getInputData import getInputData, getHouse
from pspelt.helperFunctions.logging import Logger
from pspelt.obstacleFunctions import overlap_solids
from pspelt.helperFunctions.plotting import Show2D, plot_intersection_info, print_results


### ARGUMENTS
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--name", default="3Dhouse", help="name of the simulation, appears in output directory name")
parser.add_argument("--default_device", default="cuda", type=str, help="run on cuda or cpu")
parser.add_argument("--float_dtype", default="foat32", choices=["foat32", "foat64", "single", "double", "half"], help="data type for floating point calculations in torch")
parser.add_argument("--t_sim_max", default=(72*60*60), type=float, help="max. walltime to simulate, default is 72 h for cluster use. sim stops at 0.99*t_max_sim")  # andere max.Zeit? wie lange braucht das "drum rum"? kann cih simulation auch die scho vergangene Zeit übergeben? dann kann ich mit nem größeren Wert rechnen und sim ist variabel darin, wie viel Zeit es noch hat
# bei pspelt noch "--tmax"

parser.add_argument("--cluster", action='store_true', help="")  # brauche ich das?
parser.add_argument("--outdir", default=os.getcwd(), type=str, help="directory to save output files to; vtk-files will be saved in seperate dir, if outputdir_vtk is specified")
parser.add_argument("--outdir_vtk", default=None, type=str, help="")
parser.add_argument("--vtk_fps", default=10, help="frames per second_PU for VTK output; overwritten if vtk_interval is specified")
parser.add_argument("--vtk_interval", default=0, type=int, help="how many steps between vtk-output-files; overwrites vtk_fps")
parser.add_argument("--vtk_start", default=0, type=float, help="at which percentage of t_target or n_steps the vtk-reporter starts, default is 0 (from the beginning); values from 0.0 to 1.0")



parser.add_argument("--re", default=200, type=float, help="Reynolds number")
parser.add_argument("--ma", default=0.05, type=float, help="Mach number")
# viscosity PU
# char density PU
# char velocity PU

parser.add_argument("--n_steps", default=100000, type=int, help="number of steps to simulate, overwritten by t_target, if t_target is >0, end of sim will be step_start+n_steps")
parser.add_argument("--t_target", default=0, type=float, help="time in PU to simulate, t_start will be calculated by PU/LU-conversion of step_start")
parser.add_argument("--step_start", default=0, type=int, help="stepnumber to start at. Useful if sim. is started from a checkpoint and sim-data should be concatenated later on")
parser.add_argument("--collision", default="bgk", choices=['kbc', 'bgk', 'bgk_reg'], help="collision operator (bgk, kbc, reg)")
parser.add_argument("--stencil", default="D3Q27", choices=['D2Q9', 'D3Q15', 'D3Q19', 'D3Q27'], help="stencil (D2Q9, D3Q27, D3Q19, D3Q15), dimensions will be infered from D")

# AUFLÖSUNG LU Charlength
# house
parser.add_argument("--house_length_lu", default=10, type=int, help="house length in LU")
parser.add_argument("--house_length_pu", default=1, help="house length in PU")
parser.add_argument("--house_width_pu", default=0, help="width of house in crossstream direction. If left default, it will be equal to house_length_pu")
#house_origin
parser.add_argument("--roof_angle", default=45, help="roof_angle in degree (0 to <90)")
#domain dimensions

# flow
parser.add_argument("--u_init", default=2, type=int, choices=[0, 1, 2], help="0: no velocity initially, 1: velocity one uniform, 2: velocity profile") # könnte ich noch auf Philipp und mich anpassen...und uniform durch komplett WSP ersetzen


parser.add_argument("--stencil", default="D3Q27", help="stencil (D3Q27, D3Q19, D3Q15)")
parser.add_argument("--output_vtk", default=False, help="output vtk-data with 10 fps (large!)")


parser.add_argument("--nan_reporter", default=False, help="stop simulation if NaN is detected in f field")
parser.add_argument("--EQLM", default=False, help="use equilibrium_lessMemory, saved up to 20% GPU-VRAM but takes 2% longer runtime on GPU")
parser.add_argument("--from_cpt", default=False, help="start from checkpoint. (!) provide --cpt_file path")
parser.add_argument("--cpt_file", default=None, help="path and name of cpt_file to use if --from_cpt=True")
parser.add_argument("--sim_i", default=0, type=int, help="step index of last checkpoints-step to start from for time-indexing of observables etc.")
parser.add_argument("--write_cpt", default=False, help="write checkpoint after finishing simulation")

#pspelt
parser.add_argument("--input", default='landscape_3D', type=str, help="") # NOT USED
parser.add_argument("--inputtype", default='stl', type=str, help="")
parser.add_argument("--dim", default=2, type=int, choices=[2, 3], help="")  # bei mir vom stencil abhängig
parser.add_argument("--res", default=3, type=float, help="points per meter") # bei mir von house_length_lu/PU abhängig!
parser.add_argument("--depth", default=100, type=float, help="") # not used? wird an ObstacleSurface übergeben, aber nicht genutzt...
parser.add_argument("--minz", default=63, type=float, help="")  # IST DAS LU oder PU? warhscheinlich LU...aber das ist irgendwie inkonsequent
parser.add_argument("--maxz", default=123, type=float, help="")
parser.add_argument("--interpolateres", default=None, type=float, help="") # NOT USED
parser.add_argument("--nmax", default=None, type=int, help="") # bei mir "n_steps
parser.add_argument("--withhouse", action='store_true', help="")  # bei mir nicht gebraucht
parser.add_argument("--debug", action='store_true', help="")  # wird nur an obstacleSurface übergeben, aber nicht genutzt
parser.add_argument("--landscapefwbb", action='store_true', help="") # NOT USED
parser.add_argument("--allfwbb", action='store_true', help="")  # da könnte ich was ähnliche machen wie boundary_combine... aber wie gebe ich dann an welche? FWBB/HWBB/IBB genutzt werden soll
parser.add_argument("--parallel", action='store_true', help="") # parallelisierung der Nachbarsuche, die aber nich so richtig zu funktionieren scheint. Vielleicht kann man das auf CPU parallelisieren?
parser.add_argument("--interpolatecsv", action='store_true', help="interpolate csv for fast prototyping") # NOT USED
parser.add_argument("--nout", default=200, type=int, help="") # bei mir vtk_fps bzw. vtk_interval
parser.add_argument("--i_out_min", default=0, type=int, help="output only after i_out_min") # bei mir als vtk_start drin, aber relativ
parser.add_argument("--nplot", default=500, type=int, help="") # alle wie viele Schritte geplottet wird bei Philipp
parser.add_argument("--i_plot_min", default=0, type=int, help="plot only after i_plot_min") # ab wann geplottet wird, bei Philipp
parser.add_argument("--stepsize", default=500, type=int, help="") # NOT USED
parser.add_argument("--collision_data_path", default=os.path.join(os.getcwd(), 'collision_data'), type=str, help="")  # DAS BRAUCH ICH...
parser.add_argument("--no_store_coll", action='store_true', help="") # ob coll_data gespeichert wird, oder nicht... -> ohne, wirds zwar verwendet, aber nicht gespeichert
parser.add_argument("--double_precision", action='store_true', help="") # ist bei mir als float_dtype hinterlegt (s.o.)
parser.add_argument("--recalc", action='store_true', help="recalculate collision data") # DAS BRAUCHE ICH AUCH
parser.add_argument("--notree", action='store_true', help="") # NOT USED
parser.add_argument("--vmax", default=1, type=float, help="note: estimate!")  # not: estimate! / das ist char_velocity PU
parser.add_argument("--saturation", default=0.5, type=float, help="canopy partial saturation")  # not: estimate! / NOT USED
parser.add_argument("--cut_z", default=0, type=float, help="cut at z=") # NOT USED

args = vars(parser.parse_args())


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