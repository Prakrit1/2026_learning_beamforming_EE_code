#!/bin/bash
#SBATCH -J EE_rawpow_aod0.0
#SBATCH -o EE_rawpow_aod0.0_%J.out
#SBATCH -e EE_rawpow_aod0.0_%J.err
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
# Tests the "training error range too large" hypothesis (found by user
# mid-session 2026-07-13): EVERY training job all week used
# EE_TRAIN_ERROR_BOUND=0.5 -- a uniform-random CSIT error on cos(AoD)
# sampled from [-0.5, +0.5] every step. Since cos(AoD) itself only spans
# [-1, 1], a training-time error range of +-0.5 is HALF the entire possible
# range of the quantity being perturbed. The reference paper (Schroder et
# al., Fig. 3) trains three SEPARATE models, each at a SINGLE FIXED error
# level: Delta-epsilon = 0.0, 0.025, or 0.05 -- never a wide random range,
# and never anywhere near 0.5 (10x the paper's largest training value).
# This job: Delta-epsilon_aod = 0.0 (perfect CSIT, easiest case -- if SAC
# still can't approach MMSE's ceiling here, the problem has nothing to do
# with error-robustness generalization at all).
# Uses the ALREADY-FIXED Dinkelbach reward (raw-power fix, see
# handoff_prompt_EE_evaluation.txt Finding 1 -- power penalty term is
# gradient-live now, not the old dead-zone constant) and EMA lambda
# (self-calibrating, default mode). Saves under training_name
# 'EE_dinkelbach_adaptive_lwin5000_N16K3_eta0.6_rawpow' (no aod suffix
# since error_bound=0.0 -- see EE_sac.py's error_suffix logic).
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
srun python3 src/models/EE_sac.py

wait
echo "$(date): Finished jobs!"
