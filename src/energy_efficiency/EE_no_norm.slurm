#!/bin/bash
#SBATCH -J EE_no_norm
#SBATCH -o EE_no_norm_%J.out
#SBATCH -e EE_no_norm_%J.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -t 3-14:01:00
#SBATCH -p wiss
#SBATCH --mem=16G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL



# Set environment
source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS
# Selects the 'energy_efficiency_no_normalization_fixed' reward branch in
# src/models/EE_sac.py -- see its __main__ block.
export EE_REWARD_MODE=energy_efficiency_no_normalization_fixed

cd /home/parajuli/repos/2025_learning_beamforming_rsma_code
export PYTHONPATH=/home/parajuli/repos/2025_learning_beamforming_rsma_code

srun nvidia-smi
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "EE_REWARD_MODE = $EE_REWARD_MODE"

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/models/EE_sac.py

wait
echo "$(date): Finished jobs!"
