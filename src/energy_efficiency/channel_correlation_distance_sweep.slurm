#!/bin/bash
#SBATCH -J EE_channel_corr_dist
#SBATCH -o EE_channel_corr_dist_%j.out
#SBATCH -e EE_channel_corr_dist_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-02:00:00
#SBATCH -p wiss
#SBATCH --mem=2G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Runs channel_correlation_distance_sweep.py: mean channel correlation vs.
# user distance (500-100000 m, 40 points, 1000 MC iterations, 3 user pairs
# averaged), our N16K3 scenario, disable_wiggle=True. Marks our actual
# operating point (100 km mean) on the plot. No GPU needed -- pure numpy.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2025_learning_beamforming_rsma_code
export PYTHONPATH=/home/parajuli/repos/2025_learning_beamforming_rsma_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/channel_correlation_distance_sweep.py
srun echo $HOSTNAME
wait

echo "$(date): Finished channel correlation distance sweep!"
