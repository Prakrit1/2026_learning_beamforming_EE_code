#!/bin/bash
#SBATCH -J EE_classical_sweep
#SBATCH -o EE_classical_sweep_%j.out
#SBATCH -e EE_classical_sweep_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-02:00:00
#SBATCH --gres=gpu:1
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Runs classical_baselines_error_sweep.py: MMSE/MRT/ZF rate-vs-error
# (10,000-sample Monte Carlo per point, 11 points from 0.00 to 0.10),
# combined with the two SAC checkpoints into one 5-curve comparison plot.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2025_learning_beamforming_rsma_code
export PYTHONPATH=/home/parajuli/repos/2025_learning_beamforming_rsma_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/classical_baselines_error_sweep.py
srun echo $HOSTNAME
wait

echo "$(date): Finished classical baselines error sweep!"
