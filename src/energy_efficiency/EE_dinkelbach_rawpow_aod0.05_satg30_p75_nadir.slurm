#!/bin/bash
#SBATCH -J EE_3gpp_nadir_aod0.05
#SBATCH -o EE_3gpp_nadir_aod0.05_%j.out
#SBATCH -e EE_3gpp_nadir_aod0.05_%j.err
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

# Set environment
source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS

# 3GPP-aligned nadir retrain, aod=0.05 -- part of the lwin5000/3GPP-native
# triplet alongside EE_dinkelbach_rawpow_aod{0.0,0.025}_satg30_p75_nadir.slurm.
# Trains at the TR 38.821 Set-1-aligned system parameters that are the
# config.py DEFAULTS (sat_gain_dBi=30, power_constraint_watt=75 -- no env
# override needed; see config.py comments for the Table 6.1.1.1-1 /
# EIRP-density derivation). Plain lwin5000 EMA Dinkelbach lambda mode,
# matching the doc's own documented mechanism, default (untuned) LR.
# Expected training_name: EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_satg30_p75_eta0.6_rawpow
export EE_REWARD_MODE=energy_efficiency_dinkelbach_adaptive
export EE_TRAIN_ERROR_BOUND=0.05
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
echo "$(date): Finished aod=0.05 3GPP nadir retrain!"
