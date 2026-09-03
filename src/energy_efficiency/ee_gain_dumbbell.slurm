#!/bin/bash
#SBATCH -J ee_gain_dumbbell
#SBATCH -o ee_gain_dumbbell_%j.out
#SBATCH -e ee_gain_dumbbell_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 0-00:10:00
#SBATCH -p wiss
#SBATCH --mem=2G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Two-point EE-vs-power comparison (EE policy vs. rate-only baseline) --
# see ee_gain_dumbbell.py docstring. No GPU/checkpoint needed, only reads
# two already-saved gzips: rate_power_triplet.gzip (already committed) and
# power_samples_rateonly.gzip (run power_samples_rateonly.slurm first if
# missing).

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/ee_gain_dumbbell.py
srun echo $HOSTNAME
wait

echo "$(date): Finished EE-gain dumbbell comparison!"
