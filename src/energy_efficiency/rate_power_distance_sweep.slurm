#!/bin/bash
#SBATCH -J EE_dist_sweep
#SBATCH -o EE_dist_sweep_%j.out
#SBATCH -e EE_dist_sweep_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-04:00:00
#SBATCH --gres=gpu:1
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Runs rate_power_distance_sweep.py: rate AND power measured together from
# the same Monte Carlo samples, swept across mean user distance (10/25/50 km,
# closer than the paper-matching 100 km default), for all 6 SAC/EE
# checkpoints (aod=0.0/0.025/0.05 + fairness 0.5/1.5/3.0) plus MMSE, crossed
# with both roam conventions (fixed vs. proportional) and both CSIT error
# levels (0.0/0.05) -- see rate_power_distance_sweep.py's docstring.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/rate_power_distance_sweep.py
srun echo $HOSTNAME
wait

echo "$(date): Finished rate/power distance sweep!"
