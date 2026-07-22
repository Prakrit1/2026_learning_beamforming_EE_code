#!/bin/bash
#SBATCH -J EE_rate_power_sweep
#SBATCH -o EE_rate_power_sweep_%j.out
#SBATCH -e EE_rate_power_sweep_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-01:30:00
#SBATCH --gres=gpu:1
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Runs rate_power_error_sweep.py: measures rate AND power TOGETHER from the
# same Monte Carlo samples, across the CSIT error sweep, for the three
# raw-power-fixed Dinkelbach checkpoints trained at error=0.0, 0.025, and
# 0.05 (jobs 147505/148538/147506). Produces the rate-vs-power trajectory
# plot (never possible before -- every other script measures only one of
# rate/power at a time).

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2025_learning_beamforming_rsma_code
export PYTHONPATH=/home/parajuli/repos/2025_learning_beamforming_rsma_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/rate_power_error_sweep.py
srun echo $HOSTNAME
wait

echo "$(date): Finished rate-power error sweep!"
