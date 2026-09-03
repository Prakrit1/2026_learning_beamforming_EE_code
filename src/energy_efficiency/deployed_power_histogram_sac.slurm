#!/bin/bash
#SBATCH -J deployed_power_hist
#SBATCH -o deployed_power_histogram_sac_%j.out
#SBATCH -e deployed_power_histogram_sac_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 0-00:10:00
#SBATCH -p wiss
#SBATCH --mem=2G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Histogram of the deployed EE policy's own per-channel radiated power --
# see deployed_power_histogram_sac.py docstring. No GPU/checkpoint needed:
# this only reads the per-sample power draws already saved by
# power_savings_bars_triplet.py (outputs/metrics/EE_lwin5000_3gpp_triplet/
# power_savings_bars.gzip), so it runs on CPU-only nodes in seconds. If
# that gzip is missing, run power_savings_bars_triplet.slurm first.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/deployed_power_histogram_sac.py
srun echo $HOSTNAME
wait

echo "$(date): Finished deployed-power histogram!"
