#!/bin/bash
#SBATCH --job-name=cosi-diag600
#SBATCH -p zen5_dense
#SBATCH -A zen5-dense-2026_start_046-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.log
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 GRACE_NO_UPDATE_CHECK=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=
export PATH=/sofia/projects/2026_055/miniconda/bin:$PATH
cd $SLURM_SUBMIT_DIR
python3 ../scripts_h200/weyl_600K_diag.py ../weyl_vs_T_h200/weyl_trajectories.dat
