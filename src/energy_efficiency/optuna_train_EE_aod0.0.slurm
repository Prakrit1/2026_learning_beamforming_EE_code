#!/bin/bash
#SBATCH -J optuna_EE_aod0.0
#SBATCH -o optuna_EE_aod0.0_%j.out
#SBATCH -e optuna_EE_aod0.0_%j.err
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

# One of three parallel LR searches (error=0.0/0.025/0.05, see the sibling
# .slurm files) -- replaces the earlier single search (job 143462) that ran
# under EE_TRAIN_ERROR_BOUND=0.5, now known to be 10x the reference paper's
# largest training value ("Finding 3", see handoff_prompt_EE_evaluation.txt).
# Tuning LR separately per error level makes sense since the two
# error=0.0/0.05 checkpoints already trained under default LR (4.2e-5/8.8e-6)
# showed different training dynamics (episode reward trajectories,
# convergence speed) at different error bounds -- the optimum LR is not
# obviously the same across them.
#
# Now backed by persistent SQLite storage (outputs/optuna_studies/
# energy_efficiency_dinkelbach_adaptive_aod0.0.db) and per-trial exception
# handling (see optuna_train_EE.py) -- job 143462 crashed at trial 61 with
# NO persistence, losing all 60 completed trials outright. Resubmitting
# this exact script after an interruption will RESUME from the last
# completed trial instead of starting over.
export EE_REWARD_MODE=energy_efficiency_dinkelbach_adaptive
export EE_TRAIN_ERROR_BOUND=0.0
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
