#!/bin/bash
#SBATCH -J EE_3gpp_nadir_detlam_aod0025_fair15
#SBATCH -o EE_3gpp_nadir_detlam_aod0025_fair15_%J.out
#SBATCH -e EE_3gpp_nadir_detlam_aod0025_fair15_%J.err
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

# Deterministic-lambda 3GPP NADIR training, error bound 0.025 (2 of 3: aod
# 0.0/0.025/0.05, mirroring the old-system final checkpoint trio). See
# EE_dinkelbach_rawpow_aod0.05_satg30_p75_nadir_detlam.slurm for the full
# rationale (EE_LAMBDA_DETERMINISTIC=1 fixes the noise-inflated lambda
# behind the 2026-07-28 over-transmission finding); this file differs from
# its companions in EE_TRAIN_ERROR_BOUND and in using the Optuna-tuned
# learning rates from the old-system aod=0.025 search (per user decision:
# tuned LR outperformed default for this error bound; the search was only
# ever run for 0.025, so 0.0/0.05 stay at defaults like their old finals).
# FAIRNESS COMPANION of the non-fair detlam file: identical except
# EE_FAIRNESS_WEIGHT=1.5 (the old-system rate/fairness sweet spot).
# Expected training_name: EE_dinkelbach_adaptive_aod0.025_detlam512_N16K3_satg30_p75_eta0.6_rawpow_lrc1.82e-06_lra1.17e-06_fair1.5
export EE_REWARD_MODE=energy_efficiency_dinkelbach_adaptive
export EE_TRAIN_ERROR_BOUND=0.025
export EE_LAMBDA_DETERMINISTIC=1
export EE_FAIRNESS_WEIGHT=1.5
export EE_LR_CRITIC=1.82e-06
export EE_LR_ACTOR=1.17e-06

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

# fail fast on a syntax error in the edited training script
python3 -m py_compile src/models/EE_sac.py || exit 1

srun nvidia-smi
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "EE_REWARD_MODE = $EE_REWARD_MODE"
echo "EE_TRAIN_ERROR_BOUND = $EE_TRAIN_ERROR_BOUND"
echo "EE_LAMBDA_DETERMINISTIC = $EE_LAMBDA_DETERMINISTIC"
echo "EE_FAIRNESS_WEIGHT = $EE_FAIRNESS_WEIGHT"
echo "EE_LR_CRITIC = $EE_LR_CRITIC"
echo "EE_LR_ACTOR = $EE_LR_ACTOR"

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/models/EE_sac.py

wait
echo "$(date): Finished jobs!"
