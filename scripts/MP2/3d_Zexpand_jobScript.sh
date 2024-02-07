#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1 --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --mem 185G
#SBATCH --time=71:59:00
#SBATCH --output=slurm-%j_%x.out
#SBATCH --error=slurm-%j_%x.out

python 3d_cylinder_simulation.py --re 200 --gpd 20 --dpy 19 --t_target 200 --stencil $1 --dpz $2 --name ${SLURM_JOB_NAME}
#optionally: add other parameters (collision etc.)

#mv "slurm-$SLURM_JOB_ID.out" "slurm-${SLURM_JOB_ID}_${SLURM_JOB_NAME}_Re$1_GpD$2_DpY$3_T$4.out"  # optionally: put other stuff in name (collision, boundary,...)
