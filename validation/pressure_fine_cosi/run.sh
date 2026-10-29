#!/bin/bash
#SBATCH --job-name=cosi-pressure-fine
#SBATCH -p zen4_h200
#SBATCH -A zen4-h200-2026_start_046-1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --cpus-per-task=24
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export GRACE_NO_UPDATE_CHECK=1 PYTHONUNBUFFERED=1
export PATH=/sofia/projects/2026_055/miniconda/bin:$PATH
cd $SLURM_SUBMIT_DIR
nvidia-smi -L
grace-topo-phonons --poscar POSCAR --model GRACE-2L-OAM --path GXMGRX --nkpts 40 \
    --node 0 0 0 --node 0.5 0.5 0.5 --search 16 --bands 3 3 \
    --pressure 1 2 3 4 5 6 7 8 9 10 12.5 15 17.5 20 25 30 35 40
