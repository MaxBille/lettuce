import lettuce as lt
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

import warnings
import numpy as np
from lettuce.unit import UnitConversion
from lettuce.util import append_axes
from lettuce.boundary import SlipBoundary, EquilibriumBoundaryPU, BounceBackBoundary, HalfwayBounceBackBoundary, FullwayBounceBackBoundary, EquilibriumOutletP, AntiBounceBackOutlet
from lettuce.flows.obstaclemax3D import ObstacleMax3D

import torch
import time
import datetime
import os
import shutil
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
warnings.simplefilter("ignore")

##################################################
#ARGUMENT PARSING
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--re", default=200, type=float, help="Reynolds number")
parser.add_argument("--n_steps", default=100000, type=int, help="number of steps to simulate, overwritten by t_target, if t_target is >0")
parser.add_argument("--gpd", default=20, type=int, help="number of gridpoints per diameter")
parser.add_argument("--dpy", default=19, type=int, help="domain height in diameters")
parser.add_argument("--dpz", default=1, type=float, help="domain width in diameters")
parser.add_argument("--lateral_walls", default='periodic', help="boundary condition in y direction (periodic, bounceback, slip)")
parser.add_argument("--bb_type", default='fwbb', help="bounce back algorithm (fwbb, hwbb)")
parser.add_argument("--t_target", default=0, type=float, help="time in PU to simulate")
parser.add_argument("--collision", default="bgk", help="collision operator (bgk, kbc, reg)")
parser.add_argument("--name", default="3Dcylinder", help="name of the simulation, appears in output directory name")
parser.add_argument("--stencil", default="D3Q27", help="stencil (D3Q27, D3Q19, D3Q15)")

args = vars(parser.parse_args())

##################################################
#PARAMETERS

re = args["re"]            # Reynoldszahl
Ma = 0.05           # Machzahl
n_steps = args["n_steps"]    # Schrittzahl
setup_diameter = 1  # D_PU = char_length_pu -> this defines the PU-Reference
flow_velocity = 1   # U_PU = char_velocity_pu -> this defines the PU-Reference velocity (u_max or u_mean of inflow)

periodic_start = 0.9  # relative starting point of peak_finding for Cd_mean Measurement to cut of any transients

gridpoints_per_diameter = args["gpd"]  # gp_per_D -> this defines the resolution ( D_LU = GPD+1)
domain_height_in_D = args["dpy"]  # D/Y  -> this defines the domain-size and total number of Lattice-Nodes
domain_length_in_D = 2 * domain_height_in_D  # D/X = domain length in X- / flow-direction
domain_width_in_D = args["dpz"]  # D/Z = DpZ = diameters per domain width in Z-direction -> domain size in periodic 3rd dimension

# if DpY is even, resulting GPD can't be odd for symmetrical cylinder and channel
# ...if DpY is even, GPD will be corrected to be even for symemtrical cylinder
# ...use odd DpY to use odd GPD
gpd_correction=False
if domain_height_in_D % 2 == 0 and gridpoints_per_diameter % 2 != 0:
    gpd_correction = True   # gpd_was_corrected-flag
    gpd_setup = gridpoints_per_diameter   # store old gpd for output
    gridpoints_per_diameter = int(gridpoints_per_diameter/2)*2   # make gpd even
    print("(!) domain_height_in_D (DpY) is even, gridpoints_per_diameter will be "+str(gridpoints_per_diameter)+". Use odd domain_height_in_D (DpY) to enable use of odd GPD!")


# OVERWRITE n_steps, if t_target is given
T_target = 140
if args["t_target"] > 0:
    T_target = args["t_target"]
    n_steps = int(T_target*((gridpoints_per_diameter+1)/setup_diameter)*(flow_velocity/(Ma*1/np.sqrt(3))))

# SIMULATOR settings
u_init = 0    # initial velocity field: # 0: uniform u=0, # 1: uniform u=1, # 2: parabolic, amplitude u_char_lu (similar to poiseuille-flow)
perturb_init = True   # perturb initial symmetry by small sine-wave in initial velocity field -> triggers Karman-vortex street for Re > 46#
lateral_walls=args["lateral_walls"]  # type of top/bottom boundary: 'bounceback' = frictious wall, 'periodic' = periodic boundary, 'slip' = non-frictious wall
bb_type=args["bb_type"]  # choose algorithm for bounceback-boundaries: fullway 'fwbb' or halfway 'hwbb'
cylinder = True    # obstacle: True = cylinder, False = no obstacle
vtk_fps = 10    # FramesPerSecond (PU) for vtk-output

gridpoints = gridpoints_per_diameter**3*domain_length_in_D*domain_height_in_D*domain_width_in_D # calc. total number of gridpoints

##################################################
# DATA OUTPUT SETTINGS (observables, stats and vtk)

output_data = True  # output/log parameters, observables and vtk or vti (if output_vtk=True)
output_vtk = False   # vtk-output. is overwritten by output_data=False (see below)

# naming: specify batch number, batch name, version name/number and parameters to put in directory- and datafile-names
name = args["name"]

if output_data:  # toggle output of parameters, observables and vti/vtk files
    # (see above) output_vtk = True    # vtk-reporter outputs vtk or vti files for visualization in Paraview

    timestamp = datetime.datetime.now()
    timestamp = timestamp.strftime("%y%m%d")+"_"+timestamp.strftime("%H%M%S")

    # specify output directory/path
    #output_path = "/mnt/ScratchHDD1/Max_Scratch/lbm_simulations"  # lokal HBRS
    #output_path = "/home/max/Documents/lbm_simulations"  # lokal Bonn
    output_path = "/home/mbille3s/02_lbm_simulations"  # cluster HBRS
    dir_name = "/data_" + str(timestamp) + "_" + name  # create directory name for all outputs to be saved in
    os.makedirs(output_path+dir_name)
    
    vtk_path = output_path+dir_name+"/vtk/out"  # subdirectory for vtk/vti output
    print("dir_name: "+dir_name)
    if output_vtk:
        print("vtk_path: " + vtk_path)
else:  # "no output" suppresses the vtk output too
    output_vtk = False

##################################################
#SIM SETUP (instatiate objects, calculate&place obstacle, append reporters)

# lattice
if args["stencil"] == "D3Q27":
    stencil = lt.D3Q27
elif args["stencil"] == "D3Q19":
    stencil = lt.D3Q19
else:  # stencil == D3Q15
    stencil = lt.D3Q15

lattice = lt.Lattice(stencil, "cuda:0", dtype=torch.float64)

# flow
flow = ObstacleMax3D(reynolds_number=re, mach_number=Ma,
                   lattice=lattice, 
                   char_length_pu=setup_diameter, 
                   char_length_lu=gridpoints_per_diameter+1, 
                   char_velocity_pu=flow_velocity,
                   x_lu=domain_length_in_D*gridpoints_per_diameter,
                   y_lu=domain_height_in_D*gridpoints_per_diameter,
                   z_lu=domain_width_in_D*gridpoints_per_diameter,
                   lateral_walls=lateral_walls,
                   bb_type=bb_type,
                   perturb_init=perturb_init, 
                   u_init=u_init
                  )

# OBSTACLE
# define a Cylinder-Obstacle
radius_LU = 0.5 * gridpoints_per_diameter
y_pos_LU = 0.5 * gridpoints_per_diameter * domain_height_in_D + 0.5
x_pos_LU = y_pos_LU

xyz = tuple(np.linspace(1, n, n) for n in flow.shape)  #  tuple of lists (1-n, not zero-based here)
xLU, yLU, zLU = np.meshgrid(*xyz, indexing='ij')  # meshgrid of x-, y- (and z-) indexes

condition = np.sqrt((xLU - x_pos_LU) ** 2 + (yLU - y_pos_LU) ** 2) < radius_LU  # circle inequation
flow.obstacle_mask[np.where(condition)] = 1  # write obstacle_mask in flow
    
### Simulation-Object (Simulator/solver) and additional settings (tau, collision operator)
tau = flow.units.relaxation_parameter_lu

# collision operator
if args["collision"] == "kbc":
    collision = lt.KBCCollision2D(lattice,tau)
    collision_choice ="kbc"
elif args["collision"] == "reg":
    collision = lt.RegularizedCollision(lattice, tau)
    collision_choice ="reg"
else:
    collision = lt.BGKCollision(lattice, tau)
    collision_choice ="bgk"

# solver
sim = lt.Simulation(flow, lattice, 
                    collision,
                    # lt.BGKCollision(lattice, tau),
                    # lt.RegularizedCollision(lattice, tau), 
                    # lt.KBCCollision2D(lattice,tau),
                    lt.StandardStreaming(lattice)
                    )
### Reporter

# VTK Reporter -> visualization
if output_vtk == True:
    VTKreport = lt.VTKReporter(lattice, flow, interval=int(flow.units.convert_time_to_lu(1/vtk_fps)), filename_base=vtk_path)
    sim.reporters.append(VTKreport)

# Observable reporter: drag coefficient
DragObservable = lt.DragCoefficient(lattice, flow, sim._boundaries[-1],area=setup_diameter*flow.units.convert_length_to_pu(gridpoints_per_diameter*domain_width_in_D))  # create observable // ! area A=2*r is in PU
Dragreport = lt.ObservableReporter(DragObservable, out=None)  # create reporter and link to created observable
sim.reporters.append(Dragreport)  # append reporter to reporter-list of simulator/solver
    
# Observable reporter: lift coefficient
LiftObservable = lt.LiftCoefficient(lattice, flow, sim._boundaries[-1],area=setup_diameter*flow.units.convert_length_to_pu(gridpoints_per_diameter*domain_width_in_D))
Liftreport = lt.ObservableReporter(LiftObservable, out=None)
sim.reporters.append(Liftreport)

##################################################
#PRINT PARAMETERS prior to simulation:
print("shape_LU:", gridpoints_per_diameter*domain_length_in_D, "x", gridpoints_per_diameter*domain_height_in_D, "x", gridpoints_per_diameter*domain_width_in_D)
print("T with", n_steps, "steps:", round(n_steps * (setup_diameter/(gridpoints_per_diameter+1))*(Ma*1/np.sqrt(3)/flow_velocity),2), "seconds")
print("n_steps to simulate 1 second:", round(((gridpoints_per_diameter+1)/setup_diameter)*(flow_velocity/(Ma*1/np.sqrt(3))),2), "steps")
print("n_steps to simulate",T_target,"seconds:",T_target*round(((gridpoints_per_diameter+1)/setup_diameter)*(flow_velocity/(Ma*1/np.sqrt(3))),2), "steps")
if output_vtk:
    print("generates approx.", int(vtk_fps*(n_steps * (setup_diameter/(gridpoints_per_diameter+1))*(Ma*1/np.sqrt(3)/flow_velocity)))+1, ".vti/.vtk-frames")

##################################################
# RUN SIMULATION

t_start=time.time()

mlups = sim.step(int(n_steps)) #Simulation mit Schrittzahl n_steps

t_end=time.time()
runtime=t_end-t_start
# output stats
print("MLUPS:", mlups)
print("PU-Time: ",flow.units.convert_time_to_pu(n_steps)," seconds")
print("number of steps:",n_steps)
print("runtime: ",runtime, "seconds (", round(runtime/60,2),"minutes )")

##################################################
# CREATE OBSERVABLE-PLOTS & SAVE OBSERVABLE-timeseries

# DRAG COEFFICIENT
drag_coefficient = np.array(Dragreport.out)
fig, ax = plt.subplots(constrained_layout=True)
ax.plot(drag_coefficient[:,1],drag_coefficient[:,2])
ax.set_xlabel("physical time / s")
ax.set_ylabel("Coefficient of Drag Cd")
ax.set_ylim([0.5, 2.0])  # change y-limits
secax = ax.secondary_xaxis('top', functions=(flow.units.convert_time_to_lu, flow.units.convert_time_to_pu))
secax.set_xlabel("timesteps (simulation time / LU)")
if output_data:
    plt.savefig(output_path+dir_name+"/drag_coefficient.png")
    np.savetxt(output_path+dir_name+"/drag_coefficient.txt", drag_coefficient, header="stepLU  |  timePU  |  Cd  FROM str(timestamp)")

# peak finder: try calculating the mean drag coefficient from an integer number of periods, if a clear periodic signal is found
try:
    values = drag_coefficient[int(drag_coefficient.shape[0]*periodic_start-1):,2]

    peaks_max = find_peaks(values, prominence=((values.max()-values.min())/2))
    peaks_min = find_peaks(-values, prominence=((values.max()-values.min())/2))
    if peaks_min[0].shape[0] - peaks_max[0].shape[0] > 0:
        peak_number = peaks_max[0].shape[0]
    else:
        peak_number = peaks_min[0].shape[0]

    if peaks_min[0][0] < peaks_max[0][0]:
        first_peak = peaks_min[0][0]
        last_peak = peaks_max[0][peak_number-1]
    else:
        first_peak = peaks_max[0][0]
        last_peak = peaks_min[0][peak_number-1]

    drag_mean = values[first_peak:last_peak].mean()
    drag_mean_simple = values.mean()

    print("Cd, simple mean:     ",drag_mean_simple)
    print("Cd, peak_finder mean:",drag_mean)

    drag_stepsLU = drag_coefficient[int(drag_coefficient.shape[0]*periodic_start-1):,0]
    peak_max_y = values[peaks_max[0]]
    peak_max_x = drag_stepsLU[peaks_max[0]]
    peak_min_y = values[peaks_min[0]]
    peak_min_x = drag_stepsLU[peaks_min[0]]

    plt.plot(drag_stepsLU, values)
    plt.scatter(peak_max_x[:peak_number],peak_max_y[:peak_number])
    plt.scatter(peak_min_x[:peak_number],peak_min_y[:peak_number])
    plt.scatter(drag_stepsLU[first_peak],values[first_peak])
    plt.scatter(drag_stepsLU[last_peak],values[last_peak])
    plt.savefig(output_path+dir_name+"/drag_coefficient_peakfinder.png")
    peakfinder=True
except: # if signal is not sinusoidal enough, calculate only simple mean value
    print("peak-finding didn't work... probably no significant peaks visible (Re<46?), or periodic region not reached (T too small)")
    values = drag_coefficient[int(drag_coefficient.shape[0]*periodic_start-1):,2]
    drag_mean_simple = values.mean()
    peakfinder=False
    print("Cd, simple mean:",drag_mean_simple)

# LIFT COEFFICIENT
lift_coefficient = np.array(Liftreport.out)
fig, ax = plt.subplots(constrained_layout=True)
ax.plot(lift_coefficient[:,1],lift_coefficient[:,2])
ax.set_xlabel("physical time / s")
ax.set_ylabel("Coefficient of Lift Cl")
ax.set_ylim([-1.1,1.1])

secax = ax.secondary_xaxis('top', functions=(flow.units.convert_time_to_lu, flow.units.convert_time_to_pu))
secax.set_xlabel("timesteps (simulation time / LU)")
if output_data:
    plt.savefig(output_path+dir_name+"/lift_coefficient.png")
    np.savetxt(output_path+dir_name+"/lift_coefficient.txt", lift_coefficient, header="stepLU  |  timePU  |  Cl  FROM str(timestamp)")
Cl_min = lift_coefficient[int(lift_coefficient[:,2].shape[0]*0.5):,2].min()
Cl_max = lift_coefficient[int(lift_coefficient[:,2].shape[0]*0.5):,2].max()
print("Cl_peaks: \nmin", Cl_min,"\nmax", Cl_max)

# plot DRAG and LIFT together:
fig, ax = plt.subplots(layout="constrained")
drag_ax = ax.plot(drag_coefficient[:,1],drag_coefficient[:,2], color="tab:blue", label="Drag")
ax.set_xlabel("physical time / s")
ax.set_ylabel("Coefficient of Drag Cd")
ax.set_ylim([0.5,2.0])

secax = ax.secondary_xaxis('top', functions=(flow.units.convert_time_to_lu, flow.units.convert_time_to_pu))
secax.set_xlabel("timesteps (simulation time / LU)")

ax2 = ax.twinx()
lift_ax = ax2.plot(lift_coefficient[:,1],lift_coefficient[:,2], color="tab:orange", label="Lift")
ax2.set_ylabel("Coefficient of Lift Cl")
ax2.set_ylim([-1.1,1.1])


fig.legend(loc="upper left", bbox_to_anchor=(0,1), bbox_transform=ax.transAxes)

if output_data:
    plt.savefig(output_path+dir_name+"/dragAndLift_coefficient.png")

# STROUHAL number: (only makes sense for Re>46 and if periodic state is reached)
try:
    ### prototyped fft for frequency detection and calculation of strouhal-number
    # ! Drag_frequency is 2* Strouhal-Freq. Lift-freq. is Strouhal-Freq.

    X = np.fft.fft(lift_coefficient[:,2])   # fft result (amplitudes)
    N = len(X)  # number of freqs
    n = np.arange(N)   # freq index
    T = N*flow.units.convert_time_to_pu(1)   # total time measured (T_PU)
    freq = n/T   # frequencies (x-axis of spectrum)

    plt.figure
    plt.stem(freq, np.abs(X), 'b', markerfmt=" ", basefmt="-b")   # plot spectrum |X|(f)
    plt.xlabel("Freq (Hz)")
    plt.ylabel("FFT Amplitude |X(freq)|")
    plt.xlim(0,1)
    #print("max. Amplitude np.abx(X).max():", np.abs(X).max())   # for debugging
    plt.ylim(0,np.abs(X[:int(X.shape[0]*0.5)]).max())   # ylim, where highes peak is on left half of full spectrum

    if output_data:
        plt.savefig(output_path+dir_name+"/fft_Cl.png")

    freq_res = freq[1]-freq[0]   # frequency-resolution
    X_abs = np.abs(X[:int(X.shape[0]*0.4)])   # get |X| Amplitude for left half of full spectrum
    freq_peak = freq[np.argmax(X_abs)]    # find frequency with highest amplitude
    print("Frequency Peak:", freq_peak, "+-", freq_res, "Hz")
    # f = Strouhal for St=f*D/U and D=U=1 in PU
except:
    print("fft for Strouhal didn't work")
    freq_res = 0
    freq_peak = 0

##################################################
# OUTPUT DATA and stats to directory

# output data
if output_data:
    output_file = open(output_path+dir_name+"/"+timestamp + "_parameters_and_observables.txt", "a")
    output_file.write("DATA for "+timestamp)
    output_file.write("\n\n###################\n\nSIM-Parameters")
    output_file.write("\nRe = "+str(re))
    output_file.write("\nMa = "+str(Ma))
    output_file.write("\nn_steps = "+str(n_steps))
    output_file.write("\nsetup_diameter (D_PU) = "+str(setup_diameter))
    output_file.write("\nflow_velocity (U_PU) = "+str(flow_velocity))
    output_file.write("\nstencil = "+str(args["stencil"]))
    output_file.write("\ngridpoints_per_diameter (gpd) = "+str(gridpoints_per_diameter))
    if gpd_correction:
        output_file.write("\ngpd was corrected from: "+str(gpd_setup)+" to "+str(gridpoints_per_diameter)+" because D/Y is even")
    output_file.write("\ndomain_length_in_D (D/X) = " + str(domain_length_in_D))
    output_file.write("\ndomain_height_in_D (D/Y) = "+str(domain_height_in_D))
    output_file.write("\ndomain_width_in_D (D/Z) = "+str(domain_width_in_D))
    output_file.write("\n")
    output_file.write("\nu_init = " + str(u_init))
    output_file.write("\nperturb_init = " + str(perturb_init))
    output_file.write("\nlateral_walls = " + str(lateral_walls))
    output_file.write("\ncollision = " + str(collision_choice))
    output_file.write("\nbb_type = " + str(bb_type))
    output_file.write("\nvtk_fps = " + str(vtk_fps))
    output_file.write("\noutput_vtk = " + str(output_vtk))
    output_file.write("\n")
    output_file.write("\nshape_LU: "+ str(flow.shape[0]) + " x " + str(flow.shape[1]) + " x " + str(flow.shape[2]))
    output_file.write("\ntotal No. of gridpoints: "+ str(gridpoints))
    output_file.write("\n")
    output_file.write("output_dir: "+str(output_path+dir_name))
    output_file.write("\n")
    output_file.write("\n###################\n\ncylinder:")
    output_file.write("\nradius_LU = "+str(radius_LU))
    output_file.write("\nx_pos_LU = "+str(x_pos_LU))
    output_file.write("\ny_pos_LU = "+str(y_pos_LU))
    output_file.write("\ntau = "+str(tau))
    output_file.write("\n")
    output_file.write("\n###################\n\nSTATS:")
    output_file.write("\nT_PU = "+str(flow.units.convert_time_to_pu(n_steps))+" seconds")
    output_file.write("\nruntime = "+str(runtime)+ " seconds (="+str(runtime/60)+" minutes)")
    output_file.write("\nMLUPS = "+str(mlups))
    output_file.write("\n")
    output_file.write("\n###################\n\nOBSERVABLES:")
    output_file.write("\nCoefficient of drag between "+str(round(drag_coefficient[int(drag_coefficient.shape[0]*periodic_start-1),1],2))+" s and "+str(round(drag_coefficient[int(drag_coefficient.shape[0]-1),1],2))+" s:")
    output_file.write("\nCd_mean, simple      = "+str(drag_mean_simple))
    if peakfinder:
        output_file.write("\nCd_mean, peak_finder = "+str(drag_mean))
    else:
        output_file.write("\nnoPeaksFound")
    output_file.write("\nCd_min = "+str(drag_coefficient[int(drag_coefficient.shape[0]*periodic_start-1):,2].min()))
    output_file.write("\nCd_max = "+str(drag_coefficient[int(drag_coefficient.shape[0]*periodic_start-1):,2].max()))
    output_file.write("\n")
    output_file.write("\nCoefficient of lift:")
    output_file.write("\nCl_min = "+str(Cl_min))
    output_file.write("\nCl_max = "+str(Cl_max))
    output_file.write("\n")
    output_file.write("\nStrouhal number:")
    output_file.write("\nf +- df = "+str(freq_peak)+" +- "+str(freq_res)+" Hz")
    output_file.close()

#output copyable numbers for EXCEL etc.
if output_data:
    output_file = open(output_path+dir_name+"/"+timestamp + "_parameters_and_observables_copyable.txt", "a")
    output_file.write("DATA for "+timestamp)
    output_file.write("\n\n###################\n\nSIM-Parameters: Re, Ma, n_steps, setup_diameter, flow_velocity,GPD, DpY, DpX,u_init, perturb_init, bb_wall, vtk_fps, output_vtk, shape_LU, gridpoints, output_dir, radius_LU, x_pos_LU, y_pos_LU, tau, T_PU, runtime, MLUPS")
    output_file.write("\n"+str(re))
    output_file.write("\n"+str(Ma))
    output_file.write("\n"+str(n_steps))
    output_file.write("\n"+str(setup_diameter))
    output_file.write("\n"+str(flow_velocity))
    output_file.write("\n"+str(gridpoints_per_diameter))
    output_file.write("\n" + str(domain_length_in_D))
    output_file.write("\n"+str(domain_height_in_D))
    output_file.write("\n"+str(domain_width_in_D))
    output_file.write("\n"+str(u_init))
    output_file.write("\n"+str(perturb_init))
    output_file.write("\n"+str(lateral_walls))
    output_file.write("\n"+str(collision_choice))
    output_file.write("\n"+str(bb_type))
    output_file.write("\n"+str(vtk_fps))
    output_file.write("\n"+str(output_vtk))
    output_file.write("\n")
    output_file.write("\n"+ str(flow.shape[0]) + " x " + str(flow.shape[1]) + " x " + str(flow.shape[2]))
    output_file.write("\n"+ str(gridpoints))
    output_file.write("\n")
    output_file.write(""+str(output_path+dir_name))
    output_file.write("\n")
    output_file.write("\n"+str(radius_LU))
    output_file.write("\n"+str(x_pos_LU))
    output_file.write("\n"+str(y_pos_LU))
    output_file.write("\n"+str(tau))
    output_file.write("\n")
    output_file.write("\n"+str(flow.units.convert_time_to_pu(n_steps)))
    output_file.write("\n"+str(runtime))
    output_file.write("\n"+str(mlups))
    output_file.write("\n")
    output_file.write("\n###################\n\nOBSERVABLES: CdmeanSimple, (Cdpeakfinder), Cdmin,Cdmax,Clmin,Clmax,St,df")
    output_file.write("\nCoefficient of drag between "+str(round(drag_coefficient[int(drag_coefficient.shape[0]*periodic_start-1),1],2))+" s and "+str(round(drag_coefficient[int(drag_coefficient.shape[0]-1),1],2))+" s:")
    output_file.write("\n"+str(drag_mean_simple))
    if peakfinder:
        output_file.write("\n"+str(drag_mean))
    else:
        output_file.write("\nnoPeaksFound")
    output_file.write("\n"+str(drag_coefficient[int(drag_coefficient.shape[0]*periodic_start-1):,2].min()))
    output_file.write("\n"+str(drag_coefficient[int(drag_coefficient.shape[0]*periodic_start-1):,2].max()))
    output_file.write("\n")
    output_file.write("\n"+str(Cl_min))
    output_file.write("\n"+str(Cl_max))
    output_file.write("\n")
    output_file.write("\n"+str(freq_peak))
    output_file.write("\n"+str(freq_res))
    output_file.close()

### CUDA-VRAM-summary:
    output_file = open(output_path+dir_name+"/"+timestamp + "_GPU_memory_summary.txt", "a")
    output_file.write("DATA for "+timestamp+"\n\n")
    output_file.write(torch.cuda.memory_summary(device="cuda:0"))
    output_file.close()
    
### list present torch tensors: (for memory troubleshooting/analysis)
    output_file = open(output_path+dir_name+"/"+timestamp + "_GPU_list_of_tensors.txt", "a")
    total_bytes = 0
    import gc
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj,'data') and torch.is_tensor(obj.data)):
                output_file.write("\n"+str(obj.size())+", "+str(obj.nelement()*obj.element_size()))
                total_bytes = total_bytes+obj.nelement()*obj.element_size()    
        except:
            pass
    output_file.write("\n\ntotal bytes for tensors:"+str(total_bytes))
    output_file.close()

### count occurence of tensors in list of tensors: (for memory troubleshooting/analysis)
    from collections import Counter
    my_file = open(output_path+dir_name+"/"+timestamp + "_GPU_list_of_tensors.txt","r")
    data=my_file.read()
    my_file.close()
    data_into_list=data.split("\n")
    c = Counter(data_into_list)
    output_file = open(output_path+dir_name+"/"+timestamp + "_GPU_counted_tensors.txt", "a")
    for k,v in c.items():
        output_file.write("type,size,bytes: {}, number: {}\n".format(k,v) )
    output_file.write("\ntotal bytes for tensors:"+str(total_bytes))
    output_file.close()
    