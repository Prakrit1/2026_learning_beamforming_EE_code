#!/bin/bash
#SBATCH -J SAC_rateonly_aod0.025_nadir
#SBATCH -o SAC_rateonly_aod0.025_nadir_%j.out
#SBATCH -e SAC_rateonly_aod0.025_nadir_%j.err
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

# Expected training_name: SAC_rateonly_aod0.025_N16K3_satg30_p75_eta0.6_rawpow

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS

export EE_REWARD_MODE=sum_rate_only
export EE_TRAIN_ERROR_BOUND=0.025
export EE_POWER_BUDGET_WATT=75

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

python3 -m py_compile src/models/algorithms/soft_actor_critic.py src/models/EE_sac.py || exit 1

srun nvidia-smi
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "EE_REWARD_MODE = $EE_REWARD_MODE"
echo "EE_TRAIN_ERROR_BOUND = $EE_TRAIN_ERROR_BOUND"
echo "EE_POWER_BUDGET_WATT = $EE_POWER_BUDGET_WATT"

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/models/EE_sac.py
srun echo $HOSTNAME

wait
echo "$(date): Finished SAC rate-only aod0.025 nadir training!"
