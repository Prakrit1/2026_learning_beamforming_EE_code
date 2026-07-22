#!/bin/bash
#SBATCH -J EE_classical_baselines
#SBATCH -o EE_classical_baselines_%j.out
#SBATCH -e EE_classical_baselines_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-01:00:00
#SBATCH --gres=gpu:1
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Runs classical_baselines_zero_error.py: MMSE/MRT/ZF rate at zero CSIT
# error (10,000-sample Monte Carlo each), plus the extended power bar chart
# and rate comparison chart against the two SAC checkpoints.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2025_learning_beamforming_rsma_code
export PYTHONPATH=/home/parajuli/repos/2025_learning_beamforming_rsma_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/classical_baselines_zero_error.py
srun echo $HOSTNAME
wait

echo "$(date): Finished classical baselines comparison!"
