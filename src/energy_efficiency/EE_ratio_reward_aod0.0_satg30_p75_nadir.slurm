#!/bin/bash
#SBATCH -J EE_3gpp_nadir_ratio_aod0.0
#SBATCH -o EE_3gpp_nadir_ratio_aod0.0_%j.out
#SBATCH -e EE_3gpp_nadir_ratio_aod0.0_%j.err
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

# Raw-ratio-reward ablation vs the Dinkelbach-adaptive-lambda working line
# (EE_dinkelbach_rawpow_aod{0.0,0.035,0.05}_satg30_p75_nadir.slurm). Uses
# EE_REWARD_MODE=energy_efficiency (Scheme I: reward = sum_rate/total_power
# computed directly every step, precoder always rescaled to the full power
# budget -- no lambda, no adaptive pricing at all). Same 3GPP Set-1 system
# parameters as the lambda triplet (sat_gain_dBi=30, power_constraint_watt=75
# -- config.py defaults, no override needed).
# Expected training_name: full_EE_N16K3_satg30_p75_eta0.6

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS

export EE_REWARD_MODE=energy_efficiency
export EE_TRAIN_ERROR_BOUND=0.0

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

python3 -m py_compile src/models/algorithms/soft_actor_critic.py src/models/EE_sac.py || exit 1

srun nvidia-smi
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "EE_REWARD_MODE = $EE_REWARD_MODE"
echo "EE_TRAIN_ERROR_BOUND = $EE_TRAIN_ERROR_BOUND"

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/models/EE_sac.py
srun echo $HOSTNAME

wait
echo "$(date): Finished aod=0.0 3GPP nadir ratio-reward ablation!"
