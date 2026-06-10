#!/bin/bash
#SBATCH -J EE
#SBATCH -o EE_%J.out
#SBATCH -e EE_%J.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH -t 3-14:01:00
#SBATCH -p wiss
#SBATCH --mem=16G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL
#SBATCH --nodelist=fuchu

# Set environment
source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming        # <-- changed this

export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS

cd /home/parajuli/repos/2025_learning_beamforming_rsma_code
export PYTHONPATH=/home/parajuli/repos/2025_learning_beamforming_rsma_code

srun nvidia-smi
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/models/train_sac_energy_effiency.py

wait
echo "$(date): Finished jobs!"
