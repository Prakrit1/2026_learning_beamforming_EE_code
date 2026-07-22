#!/bin/bash
#SBATCH -J optuna_EE_aod0.025
#SBATCH -o optuna_EE_aod0.025_%j.out
#SBATCH -e optuna_EE_aod0.025_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -t 9-00:00:00
#SBATCH -p wiss
#SBATCH --mem=16G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Set environment
source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS

# One of three parallel LR searches (error=0.0/0.025/0.05) -- see
# optuna_train_EE_aod0.0.slurm's comment for the full rationale. This one:
# Delta-epsilon_aod=0.025, the reference paper's MIDDLE training point
# (Fig. 3: 0.0/0.025/0.05) -- not yet trained/evaluated in this project at
# all, so this search doubles as generating the first LR-tuned checkpoint
# at exactly this error level once the best trial's params are used for a
# full training run afterward.
export EE_REWARD_MODE=energy_efficiency_dinkelbach_adaptive
export EE_TRAIN_ERROR_BOUND=0.025
export EE_DINKELBACH_LAMBDA_WINDOW_STEPS=5000

cd /home/parajuli/repos/2025_learning_beamforming_rsma_code
export PYTHONPATH=/home/parajuli/repos/2025_learning_beamforming_rsma_code

srun nvidia-smi
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "EE_REWARD_MODE = $EE_REWARD_MODE"
echo "EE_TRAIN_ERROR_BOUND = $EE_TRAIN_ERROR_BOUND"
echo "EE_DINKELBACH_LAMBDA_WINDOW_STEPS = $EE_DINKELBACH_LAMBDA_WINDOW_STEPS"

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/models/optuna_train_EE.py

wait
echo "$(date): Finished jobs!"
