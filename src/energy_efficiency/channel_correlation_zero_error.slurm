#!/bin/bash
#SBATCH -J EE_channel_corr
#SBATCH -o EE_channel_corr_%j.out
#SBATCH -e EE_channel_corr_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-00:30:00
#SBATCH -p wiss
#SBATCH --mem=2G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Runs channel_correlation_zero_error.py: pairwise channel correlation
# between our K=3 users under the current N16K3/100km scenario, zero CSIT
# error, 10,000-sample Monte Carlo. No GPU needed -- pure numpy.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2025_learning_beamforming_rsma_code
export PYTHONPATH=/home/parajuli/repos/2025_learning_beamforming_rsma_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/channel_correlation_zero_error.py
srun echo $HOSTNAME
wait

echo "$(date): Finished channel correlation check!"
