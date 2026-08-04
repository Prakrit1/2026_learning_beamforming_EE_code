#!/bin/bash
#SBATCH -J EE_pcond_aod05
#SBATCH -o EE_pcond_aod05_%j.out
#SBATCH -e EE_pcond_aod05_%j.err
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

# Power-conditioned pilot (2026-07-31), side experiment run ALONGSIDE the
# ongoing Dinkelbach + entropy-annealing pilot (job 155683), not a
# replacement for it -- both approaches are being tested in parallel.
#
# Instead of pricing power inside the reward (Dinkelbach's lambda), the power
# budget is sampled fresh per transition (log-uniform in [1, 75] W) and fed
# to the actor as an extra state input; the precoder is always rescaled to
# exactly that sampled budget (Scheme I's always-rescale logic, generalized
# from a hardcoded constant to a variable input), reward is pure rate (no
# power term at all). Idea: learn the rate-maximizing frontier r*(H, P) in
# one network; picking the EE-optimal P becomes a cheap offline grid search
# over a converged, non-training network afterwards, instead of a live
# concurrently-trained multiplier -- sidesteps every lambda-mechanism
# instability issue (undertraining, rollout-noise-inflation, entropy bias,
# EMA/warm-start tuning) this project has spent multiple sessions on.
#
# Same error bound / geometry / system params as the ongoing Dinkelbach pilot
# (job 155683: aod=0.05, nadir, 3GPP defaults 30 dBi / 75 W) so the two are
# comparable later. Validated via smoke_test_power_conditioned.py (jobs
# 155695 -- caught a missing entry in the reward-key whitelist -- and 155696,
# clean pass) before this.
# Expected training_name: EE_power_conditioned_aod0.05_N16K3_satg30_p75_eta0.6_rawpow_pcond1to75log_uniform

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS

export EE_REWARD_MODE=energy_efficiency_power_conditioned
export EE_TRAIN_ERROR_BOUND=0.05
export EE_POWER_COND_MIN_WATT=1.0
export EE_POWER_COND_MAX_WATT=75.0
export EE_POWER_COND_SAMPLING=log_uniform

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

python3 -m py_compile src/models/EE_sac.py || exit 1

srun nvidia-smi
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "EE_REWARD_MODE = $EE_REWARD_MODE"
echo "EE_TRAIN_ERROR_BOUND = $EE_TRAIN_ERROR_BOUND"
echo "EE_POWER_COND_MIN_WATT = $EE_POWER_COND_MIN_WATT"
echo "EE_POWER_COND_MAX_WATT = $EE_POWER_COND_MAX_WATT"
echo "EE_POWER_COND_SAMPLING = $EE_POWER_COND_SAMPLING"

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/models/EE_sac.py
srun echo $HOSTNAME

wait
echo "$(date): Finished power-conditioned pilot!"
