#!/bin/bash
#SBATCH -J EE_mmse_mp_detlam
#SBATCH -o EE_mmse_mp_detlam_%j.out
#SBATCH -e EE_mmse_mp_detlam_%j.err
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

# Matched-power MMSE for the four detlam checkpoints, using each
# checkpoint's own measured power (from rate_power_detlam.gzip) instead of
# the full 75 W budget. Sanity check: does SAC sit close to what MMSE
# achieves at the SAME power, especially at error=0 for the aod=0.0
# checkpoint? See mmse_matched_power_detlam.py docstring.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

python3 -m py_compile src/energy_efficiency/mmse_matched_power_detlam.py

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/mmse_matched_power_detlam.py
srun echo $HOSTNAME
wait

echo "$(date): Finished matched-power MMSE for detlam checkpoints!"
