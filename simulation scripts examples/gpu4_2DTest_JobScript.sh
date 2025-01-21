#!/bin/bash
#SBATCH --partition=gpu4
#SBATCH --nodes=1
#SBATCH --exclude=wr14
#SBATCH --mem=100G
#SBATCH --gpus-per-task=1
#SBATCH --time=71:59:00
#SBATCH --output=slurm-%J_%x.out
#SBATCH --job-name=GPU_GO_a_bit_less_BRRRRRRRR

python 2d_cylinder_simulation_v5.py --re 200 --gpd 16 --dpy 10 --t_target 200 --bc_type ibb1c2 --collision bgk --calcUProfiles True --name Test2DSimulation

#optionally: add other parameters (collision etc.)
