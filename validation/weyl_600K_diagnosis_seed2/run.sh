#!/bin/bash
#SBATCH --job-name=cosi-fc600-s2
#SBATCH -p zen4_h200
#SBATCH -A zen4-h200-2026_start_046-1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --cpus-per-task=24
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK GRACE_NO_UPDATE_CHECK=1 PYTHONUNBUFFERED=1
export PATH=/sofia/projects/2026_055/miniconda/bin:$PATH
cd $SLURM_SUBMIT_DIR
python3 ../scripts_h200/fc_dump.py 600 2 || exit 1
xz -T8 -9 fc_600K_full.yaml fc_600K_a.yaml fc_600K_b.yaml
python3 ../scripts_h200/replay_track_600K.py ../weyl_vs_T_h200_fixed_seed2/weyl_trajectories.dat
