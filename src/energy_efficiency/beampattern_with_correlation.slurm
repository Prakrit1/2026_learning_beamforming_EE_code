#!/bin/bash
#SBATCH -J EE_beam_corr
#SBATCH -o EE_beam_corr_%j.out
#SBATCH -e EE_beam_corr_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-01:00:00
#SBATCH --gres=gpu:1
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Runs beampattern_with_correlation.py: regenerates the EE-vs-regular
# beampattern comparison (30 realizations, CSIT error=0.05) with pairwise
# channel correlation also captured per realization, then plots beampattern
# + correlation side by side for every realization.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2025_learning_beamforming_rsma_code
export PYTHONPATH=/home/parajuli/repos/2025_learning_beamforming_rsma_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/beampattern_with_correlation.py
srun echo $HOSTNAME
wait

echo "$(date): Finished beampattern + correlation comparison!"
