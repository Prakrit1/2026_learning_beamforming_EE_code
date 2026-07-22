#!/bin/bash
#SBATCH -J EE_pptx
#SBATCH -o EE_pptx_%j.out
#SBATCH -e EE_pptx_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 0-00:15:00
#SBATCH -p wiss
#SBATCH --mem=2G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Builds docs/EE_presentation.pptx from the scratch build script. No GPU
# needed -- pure python-pptx/matplotlib/PIL scripting -- but per project rule
# everything on this shared node goes through sbatch, no exceptions.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2025_learning_beamforming_rsma_code
export PYTHONPATH=/home/parajuli/repos/2025_learning_beamforming_rsma_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 "$1"
srun echo $HOSTNAME
wait

echo "$(date): Finished build!"
