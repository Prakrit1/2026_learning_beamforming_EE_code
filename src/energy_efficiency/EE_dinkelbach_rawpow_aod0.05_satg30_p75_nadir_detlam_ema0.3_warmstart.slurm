#!/bin/bash
#SBATCH -J EE_detlam_ema_ws_aod05
#SBATCH -o EE_detlam_ema_ws_aod05_%J.out
#SBATCH -e EE_detlam_ema_ws_aod05_%J.err
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

# Retrain of job 154762 (aod0.05, plain detlam), warm-started + EMA-smoothed.
# See EE_dinkelbach_rawpow_aod0.0_satg30_p75_nadir_detlam_ema0.3_warmstart.slurm
# for the full rationale; this file differs only in EE_TRAIN_ERROR_BOUND and
# the warm-start source (job 154762's own checkpoint). aod0.05's lambda/rate
# looked plateaued-but-noisy rather than still-rising, so here the EMA
# smoothing is the primary expected benefit, warm-start mostly a "why not"
# given the checkpoint is available.
# Expected training_name: EE_dinkelbach_adaptive_aod0.05_detlam512_ema0.3_N16K3_satg30_p75_eta0.6_rawpow_warmstart
export EE_REWARD_MODE=energy_efficiency_dinkelbach_adaptive
export EE_TRAIN_ERROR_BOUND=0.05
export EE_LAMBDA_DETERMINISTIC=1
export EE_LAMBDA_DET_EMA_ALPHA=0.3
export EE_WARM_START_MODEL_PATH=/home/parajuli/repos/2026_learning_beamforming_EE_code/models/EE_dinkelbach_adaptive_aod0.05_detlam512_N16K3_satg30_p75_eta0.6_rawpow/base/full_snap_energy_effiency_1.971

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
