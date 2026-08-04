#!/bin/bash
#SBATCH -J EE_tanneal_aod05
#SBATCH -o EE_tanneal_aod05_%j.out
#SBATCH -e EE_tanneal_aod05_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -t 3-14:01:00
#SBATCH -p wiss
#SBATCH --mem=16G
#SBATCH --exclude=funchal,monaco,montevideo,funkia
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Entropy-annealing pilot (2026-07-31), per docs/EE_formulation.tex Sec 6.3:
# the doc's own theoretical grounding for the online, reward-embedded
# Dinkelbach lambda update identifies a fixed (non-annealed) target_entropy
# as the one open gap in that argument, and prescribes annealing it as the
# fix -- NOT switching lambda's estimator away from the documented rollout-
# windowed form. This run tests that prescribed fix directly against the
# ORIGINAL "over-transmission finding" checkpoint (job 153655,
# EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_satg30_p75_eta0.6_rawpow,
# lambda plateaued ~2.09, ~40.8 W mean power, 18.6% EE loss vs. its own
# power-scaling-diagnostic optimum) -- same reward setup, same lwin5000
# lambda mode (NOT detlam, which sidesteps rollout noise by construction and
# isn't what this doc section is about), only new variable is entropy
# annealing target_entropy 1.0 -> 0.0 linearly over the full 13,000-episode
# run (EE_TARGET_ENTROPY_ANNEAL_EPISODES left unset -> defaults to the full
# run). See src/models/algorithms/soft_actor_critic.py's set_target_entropy()
# and EE_sac.py's EE_TARGET_ENTROPY_ANNEAL_END comment for the mechanism;
# validated via smoke_test_target_entropy_anneal.py (job 155670) before this.
# Expected training_name: EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_satg30_p75_eta0.6_rawpow_tanneal1to0

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS

export EE_REWARD_MODE=energy_efficiency_dinkelbach_adaptive
export EE_TRAIN_ERROR_BOUND=0.05
export EE_DINKELBACH_LAMBDA_WINDOW_STEPS=5000
export EE_TARGET_ENTROPY_ANNEAL_END=0.0

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

python3 -m py_compile src/models/algorithms/soft_actor_critic.py src/models/EE_sac.py || exit 1

srun nvidia-smi
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "EE_REWARD_MODE = $EE_REWARD_MODE"
echo "EE_TRAIN_ERROR_BOUND = $EE_TRAIN_ERROR_BOUND"
echo "EE_DINKELBACH_LAMBDA_WINDOW_STEPS = $EE_DINKELBACH_LAMBDA_WINDOW_STEPS"
echo "EE_TARGET_ENTROPY_ANNEAL_END = $EE_TARGET_ENTROPY_ANNEAL_END"

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/models/EE_sac.py
srun echo $HOSTNAME

wait
echo "$(date): Finished entropy-annealing pilot!"
