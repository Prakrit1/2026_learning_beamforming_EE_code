#!/bin/bash
#SBATCH -J SAC_rateonly_nadir
#SBATCH -o SAC_rateonly_nadir_%j.out
#SBATCH -e SAC_rateonly_nadir_%j.err
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

# Rate-only SAC baseline (schroder2025modelfree): reward = sum rate alone, no
# Dinkelbach price term. Precoder is still clip-projected (get_precoding_learned_clip_only-
# compatible), so per Lemma 1 (sum rate monotonic in scale) the policy should learn to
# push its raw output to the power budget on its own, without an always-rescale
# projection at training time. Same 3GPP Set-1 system parameters (sat_gain_dBi=30,
# power_constraint_watt=75 -- config.py defaults, no override needed) as the
# EE_dinkelbach_adaptive aod=0.0 nadir checkpoint, so the two are directly comparable
# under get_precoding_learned_clip_only at evaluation time.
# Expected training_name: SAC_rateonly_N16K3_satg30_p75_eta0.6_rawpow

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS

export EE_REWARD_MODE=sum_rate_only
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
echo "$(date): Finished SAC rate-only nadir baseline training!"
