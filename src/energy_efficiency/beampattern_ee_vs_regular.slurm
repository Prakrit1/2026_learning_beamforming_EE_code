#!/bin/bash
#SBATCH -J EE_beampattern
#SBATCH -o EE_beampattern_%j.out
#SBATCH -e EE_beampattern_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-01:00:00
#SBATCH --gres=gpu:1
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Runs beampattern_ee_vs_regular.py: generates + plots beampatterns for
# MMSE (spatial reference), the regular Scheme I SAC checkpoint, and the
# EE Dinkelbach SAC checkpoint (error=0.05-trained), all evaluated at CSIT
# error=0.05, to check whether EE and regular SAC point beams in the same
# spatial direction (just using less power) or something different.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2025_learning_beamforming_rsma_code
export PYTHONPATH=/home/parajuli/repos/2025_learning_beamforming_rsma_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/beampattern_ee_vs_regular.py --plot-only --realization ${REALIZATION_IDX:-0}
srun echo $HOSTNAME
wait

echo "$(date): Finished beampattern comparison!"
