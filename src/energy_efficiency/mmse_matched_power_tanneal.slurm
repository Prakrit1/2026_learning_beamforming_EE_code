#!/bin/bash
#SBATCH -J EE_mmse_mp_tanneal
#SBATCH -o EE_mmse_mp_tanneal_%j.out
#SBATCH -e EE_mmse_mp_tanneal_%j.err
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

# Matched-power MMSE for the entropy-anneal checkpoint, using its own
# measured power (from rate_power_tanneal.gzip) instead of the full 75 W
# budget -- the decisive test of whether entropy annealing closes the
# matched-power-MMSE gap that detlam/EMA+warmstart could not (43-66% across
# 2026-07-29/30). Needs rate_power_tanneal.py's gzip -- submit with
# --dependency=afterok:<rate_power job>. See mmse_matched_power_tanneal.py
# docstring.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

python3 -m py_compile src/energy_efficiency/mmse_matched_power_tanneal.py

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/mmse_matched_power_tanneal.py
srun echo $HOSTNAME
wait

echo "$(date): Finished matched-power MMSE for tanneal checkpoint!"
