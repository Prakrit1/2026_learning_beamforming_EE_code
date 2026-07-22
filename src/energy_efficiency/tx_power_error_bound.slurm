#!/bin/bash
#SBATCH -J EE_txpower_errbound
#SBATCH -o EE_txpower_errbound_%j.out
#SBATCH -e EE_txpower_errbound_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-01:00:00
#SBATCH --gres=gpu:1
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Runs the full canonical tx_power_distribution.py (all blocks) including
# the NEW zero-error power-distribution block for the raw-power-fixed
# Dinkelbach checkpoints trained at error=0.0 and error=0.05 (jobs
# 147505/147506). Fast -- all blocks are zero-error-only Monte Carlo, no
# error sweep.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2025_learning_beamforming_rsma_code
export PYTHONPATH=/home/parajuli/repos/2025_learning_beamforming_rsma_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/tx_power_distribution.py
srun echo $HOSTNAME
wait

echo "$(date): Finished error-bound tx power distribution!"
