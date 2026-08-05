#!/bin/bash
#SBATCH -J EE_3gpp_nadir_aod0_lambdainit
#SBATCH -o EE_3gpp_nadir_aod0_lambdainit_%j.out
#SBATCH -e EE_3gpp_nadir_aod0_lambdainit_%j.err
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

# Lambda-initialization experiment: same plain lwin5000/3GPP nadir aod=0.0
# setup as EE_dinkelbach_rawpow_aod0.0_satg30_p75_nadir.slurm (the current
# working checkpoint, job 156358), but the Dinkelbach lambda EMA is seeded
# from a closed-form prior instead of the untrained policy's first
# (essentially noise) sample. Mechanism: EE_sac.py's
# EE_DINKELBACH_LAMBDA_INIT_RATE/_POWER (new, see 04_approach.tex's
# "Reward Formulation Inspired by the Dinkelbach Transform" discussion
# paragraph) seed dinkelbach_rate_ema/dinkelbach_power_ema directly, so the
# per-step EMA blends in from a sensible starting point instead of
# hard-snapping to one noisy sample at step 1 (empirically observed to land
# near ~0.4 in prior runs, per the project handoff).
#
# Seed values below are the MMSE baseline's own mean rate/power AT THE SAME
# 75 W BUDGET, read off error_sweep_triplet_matched_power_sumrate.pdf /
# rate_power_triplet.pdf (2026-08-05 run): rate ~= 12.2 bps/Hz, power = full
# 75 W budget -> lambda_0 ~= 12.2/75 ~= 0.163. These are APPROXIMATE (read
# visually off the plotted curve, not pulled from the exact
# rate_power_triplet.gzip numbers) -- close enough for a seed the EMA will
# continue to adapt away from, but replace with the exact gzip values if
# more precision is wanted before drawing conclusions from this run.
# Expected training_name:
# EE_dinkelbach_adaptive_lwin5000_lambdainit12.2-75_N16K3_satg30_p75_eta0.6_rawpow

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS

export EE_REWARD_MODE=energy_efficiency_dinkelbach_adaptive
export EE_TRAIN_ERROR_BOUND=0.0
export EE_DINKELBACH_LAMBDA_WINDOW_STEPS=5000
export EE_DINKELBACH_LAMBDA_INIT_RATE=12.2
export EE_DINKELBACH_LAMBDA_INIT_POWER=75

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

python3 -m py_compile src/models/algorithms/soft_actor_critic.py src/models/EE_sac.py || exit 1

srun nvidia-smi
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "EE_REWARD_MODE = $EE_REWARD_MODE"
echo "EE_TRAIN_ERROR_BOUND = $EE_TRAIN_ERROR_BOUND"
echo "EE_DINKELBACH_LAMBDA_WINDOW_STEPS = $EE_DINKELBACH_LAMBDA_WINDOW_STEPS"
echo "EE_DINKELBACH_LAMBDA_INIT_RATE = $EE_DINKELBACH_LAMBDA_INIT_RATE"
echo "EE_DINKELBACH_LAMBDA_INIT_POWER = $EE_DINKELBACH_LAMBDA_INIT_POWER"

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/models/EE_sac.py
srun echo $HOSTNAME

wait
echo "$(date): Finished aod=0.0 3GPP nadir lambda-init experiment!"
