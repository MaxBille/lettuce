#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1 --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --mem 185G
#SBATCH --time=71:59:00
#SBATCH --job-name=best_jobname_ever

python 2d_cylinder_simulation.py --re $1 --gpd $2 --dpy $3 --t_target $4 # optionally: add other parameters (collision etc.)

mv "slurm-$SLURM_JOB_ID.out" "slurm-${SLURM_JOB_ID}_${SLURM_JOB_NAME}_Re$1_GpD$2_DpY$3_T$4.out"  # optionally: put other stuff in name (collision, boundary,...)
