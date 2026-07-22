#!/bin/bash
#SBATCH -J EE_rawpow_aod0.025
#SBATCH -o EE_rawpow_aod0.025_%J.out
#SBATCH -e EE_rawpow_aod0.025_%J.err
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
# Same as EE_dinkelbach_rawpow_aod0.0_N16K3.slurm's comment -- see that file
# for the full "training error range too large" hypothesis. This job:
# Delta-epsilon_aod = 0.025 -- the paper's (Schroder et al., Fig. 3) MIDDLE
# fixed training error level, completing the 0.0/0.025/0.05 three-point
# match (0.0 and 0.05 were already trained/evaluated -- see
# handoff_prompt_EE_evaluation.txt). Uses the same fixed Dinkelbach reward
# and default EMA lambda as the other two -- NOT waiting for the Optuna
# per-error-bound LR search (147529/147530/147531, still running, multi-day)
# so this checkpoint exists now; can retrain with tuned LR later if Optuna's
# result meaningfully differs.
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
srun python3 src/models/EE_sac.py

wait
echo "$(date): Finished jobs!"
