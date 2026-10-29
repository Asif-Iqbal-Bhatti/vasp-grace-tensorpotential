#!/bin/bash
#SBATCH --job-name=test-effective
#SBATCH -p zen4_h200
#SBATCH -A zen4-h200-2026_start_046-1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK GRACE_NO_UPDATE_CHECK=1 PYTHONUNBUFFERED=1
export PATH=/sofia/projects/2026_055/miniconda/bin:$PATH
cd $SLURM_SUBMIT_DIR/../..
# run only the effective_phonons snippet from tests/test_modules.py
python3 - <<'PY'
import re, textwrap, subprocess, sys, time
s = open("tests/test_modules.py").read()
code = s[s.index('effective_check = textwrap.dedent(f"""') + len('effective_check = textwrap.dedent(f"""'):]
code = textwrap.dedent(code[:code.index('""")')]).replace("{GRACE_MODEL}", "GRACE-2L-OAM")
t = time.time(); r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
print(r.stdout); print(r.stderr[-2000:] if r.returncode else ""); print(f"rc {r.returncode}, {time.time()-t:.0f} s")
PY
