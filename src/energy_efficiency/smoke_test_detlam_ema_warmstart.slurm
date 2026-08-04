#!/bin/bash
#SBATCH -J EE_smoke_detlam_ema_ws
#SBATCH -o EE_smoke_detlam_ema_ws_%j.out
#SBATCH -e EE_smoke_detlam_ema_ws_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 0-00:15:00
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Smoke test only -- 3 tiny episodes, verifies EMA-smoothed deterministic
# lambda + warm-start-from-checkpoint (both new in EE_sac.py, 2026-07-30)
# before committing to four multi-day real retrains.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

python3 -m py_compile src/models/EE_sac.py || exit 1

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/smoke_test_detlam_ema_warmstart.py
srun echo $HOSTNAME
wait

echo "$(date): Finished detlam EMA + warm-start smoke test!"
