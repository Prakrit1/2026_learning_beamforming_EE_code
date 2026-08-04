#!/bin/bash
#SBATCH -J EE_mmse_mp_detlam_emaws
#SBATCH -o EE_mmse_mp_detlam_emaws_%j.out
#SBATCH -e EE_mmse_mp_detlam_emaws_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-04:00:00
#SBATCH --gres=gpu:1
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --exclude=funchal,monaco,montevideo,funkia
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Matched-power MMSE for the four ema0.3+warmstart detlam retrains, using
# each checkpoint's own measured power (from rate_power_detlam_ema_
# warmstart.gzip) instead of the full 75 W budget. This is the direct test
# of whether ema+warmstart closed the 45-66% matched-power-MMSE gap found
# in the pre-ema detlam checkpoints. Needs rate_power_detlam_ema_warmstart.py's
# gzip -- submit with --dependency=afterok:<rate_power job>. See
# mmse_matched_power_detlam_ema_warmstart.py docstring.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

python3 -m py_compile src/energy_efficiency/mmse_matched_power_detlam_ema_warmstart.py

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/mmse_matched_power_detlam_ema_warmstart.py
srun echo $HOSTNAME
wait

echo "$(date): Finished matched-power MMSE for ema+warmstart detlam checkpoints!"
