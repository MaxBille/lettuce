README:
- the gpu4_(2D/3D)Test_JobScript.sh files are examples for slurm-batch-files to pass to the slurm scheduler on the HPC-cluster of the H-BRS. Use their python commands locally or on your preferred cluster with respective changes in the SBTACH-header.
- 3d_cylinder_simulation_v5.py is a simulation script that runs a 3D Cylinder flow and accepts parameters/arguments. Look into the code and search for argument parsing.
- without parameters, the defaults (see code) will be used.
    - for an exemplary simulation at Re3900 the following parameters can be used:
        python 3d_cylinder_simulation_v5.py --re 3900 --gpd 22 --dpy 10 --dpz 3 --t_target 200 --bc_type ibb1c2 --collision kbc --calcUProfiles True --name bestSimulationEver
    - important to choose: gpd= gridpoints per diameter (the domain size is based on this resolution, dependent onm dpy (diameters per Y-dimension), dpx and dpz
    - for Re3900 a Domain with 20x10x3 Cylinder-diameters is normally simulated. dpx = 2*dpy by default (no input needed); dpz = 3
    - target time in PU: --t_target. For t_target > 0, the parameter n_steps is ignored. If you want to use n_steps instead, don't specify t_target
    - see "#ARGUMENT PARSING" section in the script
    - BOUNCE BACK: for --bc_type <BBBC> choose the bounce back algorithm for the cylinder. 'fwbb' is the rudimentary Fullway Bounce Back boundary condition, initially implemented in lettuce 0.2.3, but modified to calculate solid fluid interaction forces through Momentum Exchange Algorithm
        - choose the variants c1, c2 for most memory- and runtime-efficient variants
    - for example: --bc_type 'ibb1c2' uses the linearly interpolated bounce back boundary condition
    - the parameter --name <NAME> will be used together with a timestamp for naming the output folder and data.

- Ma=0.05 is hardcoded and can be changed manually in the script, if needed.
- the time frame used for evaluation of Drag, Lift, Strouhal etc. is hardcoded to 0.9Tmax to Tmax for Re<1000 and 0.4Tmax to Tmax for Re>1000, with the variable "periodic_start", just try Strg+f
- the vtk-output is hardcoded at 10fps, search for "vtk_fps = 10"
- the initial velocity field and perturbation is hardcoded in the script and passed to the flow-class. see "u_init" and "perturb_init"
- for use with cuda-native lettuce-branches, you have to pass native=false to initiate Lattice class
- for compact BBBC the substep-time-measuring does not work, because GPU and CPU work asynchronous. The output of this time measurement is not usable for compact variants! Total runtime and MLUPS are still correct.

(!) change Input-/Output-paths to your correct paths, below the import section.
    - output_path: output of .txt, plots, AvgVelocity-profile data, stats etc. slurm.out is in the folder, where you started the simulation.
    - scratch_dir: vtk-output and checkpoint data. (both off by default and the folder/path is not needed)
    - diIlio_path: where the reference data in .csv-files for the AvgVelocity profiles from plots is stored.

(!) in principle the flow-class "ObstacleCylinder" is both 2D and 3D capable and infers the dimensions from "shape". At the moment the simulation script variants 3d_ and 2d_ are almost identical, apart from Stencil, Shape and dpz-Parameter and respective outputs. (will be unified in future work)
    - in the 2D script, ObstacleCylinder is instantiated with a 2D-shape
    - the 2D script doesn't contain the parameter/option to use the Equilibrium_LessMemory by M.Kliemank (normally not needed in 2D). It saves 20% VRAM and runs 2% slower if EQLM is used. If the compact variants of BBBC are used, the equilibrium becomes the memory bottleneck of simulation.
