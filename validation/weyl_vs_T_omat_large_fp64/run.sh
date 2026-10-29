#!/bin/bash
#SBATCH --job-name=cosi-weyl-omatL
#SBATCH -p zen4_h200
#SBATCH -A zen4-h200-2026_start_046-1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --cpus-per-task=24
#SBATCH --time=10:00:00
#SBATCH --output=%x-%j.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export GRACE_NO_UPDATE_CHECK=1 PYTHONUNBUFFERED=1
export PATH=/sofia/projects/2026_055/miniconda/bin:$PATH
cd $SLURM_SUBMIT_DIR
nvidia-smi -L
MODEL=GRACE-2L-OMAT-large-ft-AM-fp64
python3 ../scripts_h200/relax_cosi.py POSCAR_start_OAM $MODEL || exit 1
grace-topo-phonons --poscar POSCAR --model $MODEL --path GXMGRX --nkpts 40 \
    --search 16 --bands 3 3 --max_seeds 300 \
    --node 0 0 0 --node 0.5 0.5 0.5 \
    --temperature 100 300 600 --md_snapshots 300
