#!/bin/bash
#SBATCH -J EE_3gpp_nadir_aod0.025
#SBATCH -o EE_3gpp_nadir_aod0.025_%j.out
#SBATCH -e EE_3gpp_nadir_aod0.025_%j.err
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

# aod=0.025 counterpart to EE_dinkelbach_rawpow_aod{0.0,0.05}_satg30_p75_nadir.slurm
# (jobs 156358, 153655), completing the lwin5000/3GPP-native triplet. No prior
# aod=0.025 checkpoint exists at 3GPP params in either the old rollout-lambda
# or the abandoned detlam line (the only earlier aod=0.025 attempts were the
# pre-3GPP Optuna-LR-tuned run and a detlam run that diverged with NaN around
# episode 12k) -- this is the first aod=0.025 attempt with plain lwin5000 EMA
# Dinkelbach at 3GPP Set-1 params (sat_gain_dBi=30, power_constraint_watt=75,
# config.py defaults, no override needed). Default (untuned) LR, matching the
# aod=0.0/aod=0.05 3GPP scripts exactly -- the Optuna tuning from the old
# aod=0.025 checkpoint was for the pre-3GPP (20 dBi/100 W) system.
# Expected training_name: EE_dinkelbach_adaptive_aod0.025_lwin5000_N16K3_satg30_p75_eta0.6_rawpow

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS

export EE_REWARD_MODE=energy_efficiency_dinkelbach_adaptive
export EE_TRAIN_ERROR_BOUND=0.025
export EE_DINKELBACH_LAMBDA_WINDOW_STEPS=5000

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

python3 -m py_compile src/models/algorithms/soft_actor_critic.py src/models/EE_sac.py || exit 1

srun nvidia-smi
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "EE_REWARD_MODE = $EE_REWARD_MODE"
echo "EE_TRAIN_ERROR_BOUND = $EE_TRAIN_ERROR_BOUND"
echo "EE_DINKELBACH_LAMBDA_WINDOW_STEPS = $EE_DINKELBACH_LAMBDA_WINDOW_STEPS"

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/models/EE_sac.py
srun echo $HOSTNAME

wait
echo "$(date): Finished aod=0.025 3GPP nadir retrain!"
