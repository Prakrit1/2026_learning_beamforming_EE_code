#!/bin/bash
#SBATCH -J deployed_power_hist_compare
#SBATCH -o deployed_power_histogram_compare_%j.out
#SBATCH -e deployed_power_histogram_compare_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 0-00:10:00
#SBATCH -p wiss
#SBATCH --mem=2G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# EE-vs-rate-only deployed power histogram overlay -- see
# deployed_power_histogram_compare.py docstring. No GPU/checkpoint needed,
# only reads two already-saved gzips: run power_savings_bars_triplet.slurm
# and power_samples_rateonly.slurm first if either is missing.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/deployed_power_histogram_compare.py
srun echo $HOSTNAME
wait

echo "$(date): Finished EE-vs-rate-only power histogram comparison!"
