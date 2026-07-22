#!/bin/bash
#SBATCH -J optuna_EE_aod0.05
#SBATCH -o optuna_EE_aod0.05_%j.out
#SBATCH -e optuna_EE_aod0.05_%j.err
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
# Delta-epsilon_aod=0.05, EXACTLY matches the reference paper's largest
# single fixed training error level, and the SAME error bound already used
# for the currently-active error=0.05 checkpoint (job 147506, trained under
# the DEFAULT, un-tuned LR 4.2e-5/8.8e-6) -- this search's best trial can
# directly replace that checkpoint with an LR-tuned version for comparison.
export EE_REWARD_MODE=energy_efficiency_dinkelbach_adaptive
export EE_TRAIN_ERROR_BOUND=0.05
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
