#!/bin/bash
#SBATCH -J EE
#SBATCH -o EE_%j.out
#SBATCH -e EE_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 3-14:01:00
#SBATCH --gres=gpu:1
#SBATCH -p wiss
#SBATCH --mem=16G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL
%#SBATCH --nodelist=anatolien
# Set environment
source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming
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
