#!/bin/bash
#SBATCH -J EE_rawpow_aod0.025_lrtuned
#SBATCH -o EE_rawpow_aod0.025_lrtuned_%J.out
#SBATCH -e EE_rawpow_aod0.025_lrtuned_%J.err
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
# Retrains the aod=0.025 checkpoint (see EE_dinkelbach_rawpow_aod0.025_N16K3.slurm
# for the default-LR version, already trained/evaluated) using the best
# lr_critic/lr_actor found by the Optuna search
# (outputs/optuna_studies/energy_efficiency_dinkelbach_adaptive_aod0.025.db,
# job 147530 -- OOM-killed 2026-07-18 after 65/100 trials, not resubmitted).
# Best trial #40 (value=0.006806), found deep into the search (25 trials
# after it, none beat it): lr_critic=10^-5.739663=1.82e-6,
# lr_actor=10^-5.930565=1.18e-6.
export EE_REWARD_MODE=energy_efficiency_dinkelbach_adaptive
export EE_TRAIN_ERROR_BOUND=0.025
export EE_DINKELBACH_LAMBDA_WINDOW_STEPS=5000
export EE_LR_CRITIC=1.823e-6
export EE_LR_ACTOR=1.175e-6

cd /home/parajuli/repos/2025_learning_beamforming_rsma_code
export PYTHONPATH=/home/parajuli/repos/2025_learning_beamforming_rsma_code

srun nvidia-smi
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "EE_REWARD_MODE = $EE_REWARD_MODE"
echo "EE_TRAIN_ERROR_BOUND = $EE_TRAIN_ERROR_BOUND"
echo "EE_DINKELBACH_LAMBDA_WINDOW_STEPS = $EE_DINKELBACH_LAMBDA_WINDOW_STEPS"
echo "EE_LR_CRITIC = $EE_LR_CRITIC"
echo "EE_LR_ACTOR = $EE_LR_ACTOR"

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/models/EE_sac.py

wait
echo "$(date): Finished jobs!"
