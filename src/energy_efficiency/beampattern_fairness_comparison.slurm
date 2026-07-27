#!/bin/bash
#SBATCH -J EE_beam_fair
#SBATCH -o EE_beam_fair_%j.out
#SBATCH -e EE_beam_fair_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-01:00:00
#SBATCH --gres=gpu:1
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Generates + plots beampatterns for the aod=0.05 baseline vs. aod=0.05 +
# fairness=0.5 checkpoints (job 152699), evaluated at CSIT error=0.05, to
# see whether the Jain's-fairness reward term visibly widens the beam.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/beampattern_fairness_comparison.py --realization 0
srun echo $HOSTNAME
wait

echo "$(date): Finished fairness beampattern comparison!"
