#!/bin/bash
#SBATCH --job-name=cosi-envcheck
#SBATCH -p zen4_h200
#SBATCH -A zen4-h200-2026_start_046-1
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.log
export GRACE_NO_UPDATE_CHECK=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=24
export PATH=/sofia/projects/2026_055/miniconda/bin:$PATH
which python3
cd $SLURM_SUBMIT_DIR
nvidia-smi -L
python3 -c "import phonopy, symfc, numpy; print('phonopy', phonopy.__version__, 'symfc', symfc.__version__, 'numpy', numpy.__version__)"
python3 check_env.py
