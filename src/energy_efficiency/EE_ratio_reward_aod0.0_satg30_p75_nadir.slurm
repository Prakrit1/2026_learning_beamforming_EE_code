#!/bin/bash
#SBATCH -J EE_ratio_nadir
#SBATCH -o EE_ratio_nadir_%j.out
#SBATCH -e EE_ratio_nadir_%j.err
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

# Raw EE-ratio baseline (naive energy-efficiency maximization): reward =
# sum_rate / total_power, optimized DIRECTLY with no Dinkelbach adaptive-price
# term. This is the CLIP-ONLY, free-power variant (EE_REWARD_MODE=
# energy_efficiency_ratio) -- NOT the old removed Scheme I 'energy_efficiency'
# mode, which always rescaled to the full budget (that one behaves like
# rate-only at 75 W and never explores lower-power operating points). Here the
# precoder is clip-projected like the Dinkelbach and rate-only baselines, so
# per Lemma 1 the policy is free to fall below budget and should drive toward
# the EE-ratio maximum (~12 W, rate-starved) on its own. Expected to train
# less stably than the subtractive adaptive reward -- that instability is
# exactly the motivation for the subtractive form; keep an eye on the .err/.out
# for divergence (the run aborts on non-finite reward rather than training on
# NaN). Same 3GPP Set-1 system params (sat_gain_dBi=30, power_constraint_watt=75
# -- config.py defaults) as the EE_dinkelbach_adaptive and SAC_rateonly aod=0.0
# nadir checkpoints, so all three are directly comparable under
# get_precoding_learned_clip_only at evaluation time.
# Expected training_name: EE_ratio_N16K3_satg30_p75_eta0.6_rawpow

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS

export EE_REWARD_MODE=energy_efficiency_ratio
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
echo "$(date): Finished raw EE-ratio baseline training!"
