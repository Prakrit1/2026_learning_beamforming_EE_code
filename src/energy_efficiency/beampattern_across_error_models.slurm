#!/bin/bash
#SBATCH -J EE_beampattern_xerr
#SBATCH -o EE_beampattern_xerr_%j.out
#SBATCH -e EE_beampattern_xerr_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-01:00:00
#SBATCH --gres=gpu:1
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Runs beampattern_across_error_models.py (first, full generation -- no
# --plot-only, the gzip doesn't exist yet): generates + plots beampatterns
# for MMSE (spatial reference) and all three "final" EE checkpoints
# (error=0.0/0.025-tuned-LR/0.05), all evaluated at CSIT error=0.05, to see
# whether training at different fixed CSIT error levels changes WHERE the
# precoder points its beams, not just how gracefully it degrades.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2025_learning_beamforming_rsma_code
export PYTHONPATH=/home/parajuli/repos/2025_learning_beamforming_rsma_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/beampattern_across_error_models.py --realization 0
srun echo $HOSTNAME
wait

echo "$(date): Finished cross-error-model beampattern comparison!"
