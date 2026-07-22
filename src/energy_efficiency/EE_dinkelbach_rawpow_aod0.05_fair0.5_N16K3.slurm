#!/bin/bash
#SBATCH -J EE_rawpow_aod0.05_fair0.5
#SBATCH -o EE_rawpow_aod0.05_fair0.5_%J.out
#SBATCH -e EE_rawpow_aod0.05_fair0.5_%J.err
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
# Same base config as EE_dinkelbach_rawpow_aod0.05_N16K3.slurm (aod=0.05, our
# most "robust" checkpoint), now with the Jain's-fairness reward term
# (calc_jain_fairness, reused from train_sac.py) added on top -- see the
# beampattern robustness analysis: EE's narrow beam only covers the true
# user angle at ~54% of its own peak gain on average (vs. 72% for the
# always-100W checkpoint), and coverage ratio correlates strongly with
# achieved sum rate (r~0.92). This tests whether penalizing per-user rate
# spread widens the learned beam / stops it abandoning users, at the
# lowest of three trial weights (0.5, 1.5, 3.0 -- see the other two
# _fairX.X_ slurm scripts, submitted in parallel, no dependency between them).
export EE_REWARD_MODE=energy_efficiency_dinkelbach_adaptive
export EE_TRAIN_ERROR_BOUND=0.05
export EE_DINKELBACH_LAMBDA_WINDOW_STEPS=5000
export EE_FAIRNESS_WEIGHT=0.5

cd /home/parajuli/repos/2025_learning_beamforming_rsma_code
export PYTHONPATH=/home/parajuli/repos/2025_learning_beamforming_rsma_code

srun nvidia-smi
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "EE_REWARD_MODE = $EE_REWARD_MODE"
echo "EE_TRAIN_ERROR_BOUND = $EE_TRAIN_ERROR_BOUND"
echo "EE_DINKELBACH_LAMBDA_WINDOW_STEPS = $EE_DINKELBACH_LAMBDA_WINDOW_STEPS"
echo "EE_FAIRNESS_WEIGHT = $EE_FAIRNESS_WEIGHT"

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/models/EE_sac.py

wait
echo "$(date): Finished jobs!"
