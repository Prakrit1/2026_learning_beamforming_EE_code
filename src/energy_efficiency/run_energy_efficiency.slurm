#!/bin/bash
#SBATCH -J Example
#SBATCH -o Example.out
#SBATCH -e Example.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 3-14:01:00
gpu SBATCH --gres=gpu
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --mail-user=duee@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Set environment
source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate beamforming

# Set project root
cd /home/parajuli/repos/2025_learning_beamforming_rsma_code
export PYTHONPATH=/home/parajuli/repos/2025_learning_beamforming_rsma_code

# Execute the job
HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/models/train_sac_energy_effiency.py
srun echo $HOSTNAME
wait  
echo "$(date): Finished jobs!"
