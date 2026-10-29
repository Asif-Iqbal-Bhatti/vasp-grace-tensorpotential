#!/bin/bash
#SBATCH --job-name=cosi-Rcheck
#SBATCH -p zen5_dense
#SBATCH -A zen5-dense-2026_start_046-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK GRACE_NO_UPDATE_CHECK=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=
export PATH=/sofia/projects/2026_055/miniconda/bin:$PATH
cd $SLURM_SUBMIT_DIR
python3 ../scripts_h200/strain_R_check.py
