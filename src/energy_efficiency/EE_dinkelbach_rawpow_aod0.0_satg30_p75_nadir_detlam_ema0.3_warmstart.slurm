#!/bin/bash
#SBATCH -J EE_detlam_ema_ws_aod0
#SBATCH -o EE_detlam_ema_ws_aod0_%J.out
#SBATCH -e EE_detlam_ema_ws_aod0_%J.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -t 3-14:01:00
#SBATCH -p wiss
#SBATCH --mem=16G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL
#SBATCH --exclude=funchal,monaco,montevideo,funkia

# Set environment
source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS

# Retrain of job 154760 (aod0.0, plain detlam), warm-started + EMA-smoothed
# (2026-07-30 reconsideration of the deterministic-lambda fix). Two changes
# vs. job 154760:
#  1. EE_LAMBDA_DET_EMA_ALPHA=0.3: the per-episode deterministic lambda
#     update is now EMA-blended instead of hard-replaced, reducing
#     episode-to-episode reward-scale jitter (see EE_sac.py's
#     dinkelbach_det_ema_alpha comment).
#  2. EE_WARM_START_MODEL_PATH: seeds the policy network from job 154760's
#     best checkpoint instead of training from scratch -- that run's lambda
#     and rate were STILL RISING at episode 13000 (log analysis, 2026-07-30),
#     evidence of undertraining rather than a converged-but-poor policy.
#     Only the actor is seeded; critics/optimizer state start fresh (see
#     EE_sac.py's warm_start_model_path comment -- checkpoints only ever
#     saved actor weights).
# Expected training_name: EE_dinkelbach_adaptive_detlam512_ema0.3_N16K3_satg30_p75_eta0.6_rawpow_warmstart
export EE_REWARD_MODE=energy_efficiency_dinkelbach_adaptive
export EE_TRAIN_ERROR_BOUND=0.0
export EE_LAMBDA_DETERMINISTIC=1
export EE_LAMBDA_DET_EMA_ALPHA=0.3
export EE_WARM_START_MODEL_PATH=/home/parajuli/repos/2026_learning_beamforming_EE_code/models/EE_dinkelbach_adaptive_detlam512_N16K3_satg30_p75_eta0.6_rawpow/base/full_snap_energy_effiency_3.033

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

# fail fast on a syntax error in the edited training script
python3 -m py_compile src/models/EE_sac.py || exit 1

srun nvidia-smi
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "EE_REWARD_MODE = $EE_REWARD_MODE"
echo "EE_TRAIN_ERROR_BOUND = $EE_TRAIN_ERROR_BOUND"
echo "EE_LAMBDA_DETERMINISTIC = $EE_LAMBDA_DETERMINISTIC"
echo "EE_LAMBDA_DET_EMA_ALPHA = $EE_LAMBDA_DET_EMA_ALPHA"
echo "EE_WARM_START_MODEL_PATH = $EE_WARM_START_MODEL_PATH"

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/models/EE_sac.py

wait
echo "$(date): Finished jobs!"
