#!/bin/bash
#SBATCH -J EE_vs_power_mmse
#SBATCH -o EE_vs_transmit_power_mmse_%j.out
#SBATCH -e EE_vs_transmit_power_mmse_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-04:00:00
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# MMSE direction, EE vs. fixed transmit power sweep (1-75W, 40 points, 10k
# MC per point) at CSIT error bound 0.0 -- see ee_vs_transmit_power_sweep.py
# docstring. No GPU/checkpoint needed (MMSE is analytical), so this can run
# on CPU-only nodes. Run this BEFORE ee_vs_transmit_power_sweep_sac.slurm --
# the SAC script overlays this script's saved gzip if present.
# Pass --error X to sweep at a different CSIT error bound (edit the srun
# line below); default is 0.0.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/ee_vs_transmit_power_sweep.py
srun echo $HOSTNAME
wait

echo "$(date): Finished MMSE EE-vs-transmit-power sweep!"
