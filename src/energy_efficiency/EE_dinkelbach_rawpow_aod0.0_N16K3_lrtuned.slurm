#!/bin/bash
#SBATCH -J EE_rawpow_aod0.0_lrtuned
#SBATCH -o EE_rawpow_aod0.0_lrtuned_%J.out
#SBATCH -e EE_rawpow_aod0.0_lrtuned_%J.err
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
# Retrains the aod=0.0 checkpoint (see EE_dinkelbach_rawpow_aod0.0_N16K3.slurm
# for the default-LR version, already trained/evaluated) using the best
# lr_critic/lr_actor found by the Optuna search
# (outputs/optuna_studies/energy_efficiency_dinkelbach_adaptive_aod0.0.db,
# job 147529 -- OOM-killed 2026-07-18 after 65/100 trials, not resubmitted).
# Best trial #7 (value=0.004774): lr_critic=10^-4.383974=4.14e-5,
# lr_actor=10^-6.602540=2.50e-7. Caveat: trial 7 was found very early (of 65
# completed trials, none of the following 57 beat it) -- less converged
# evidence than the aod0.025/0.05 studies, whose bests came from deep into
# the search (trial 40/65, 60/66).
export EE_REWARD_MODE=energy_efficiency_dinkelbach_adaptive
export EE_TRAIN_ERROR_BOUND=0.0
export EE_DINKELBACH_LAMBDA_WINDOW_STEPS=5000
export EE_LR_CRITIC=4.137e-5
export EE_LR_ACTOR=2.497e-7

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
