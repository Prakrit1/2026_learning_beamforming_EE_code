#!/bin/bash
#SBATCH -J SAC_rateonly_aod0.05_p40_nadir
#SBATCH -o SAC_rateonly_aod0.05_p40_nadir_%j.out
#SBATCH -e SAC_rateonly_aod0.05_p40_nadir_%j.err
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

# Expected training_name: SAC_rateonly_aod0.05_N16K3_satg30_p40_eta0.6_rawpow
#
# Rate-only baseline retrained at a 40 W budget (not 75 W) so its beam
# direction is optimized at the SAME transmit power the triplet's matched-power
# sweep evaluates it at -- EE^{Delta-eps=0.05}'s measured operating power is
# ~40 W (essentially flat at 40 W across the whole error sweep). The old p75
# rate-only checkpoint was trained at 75 W; rescaling that 75 W-optimized
# direction down to ~40 W is off its trained operating point and made rate-only
# lose to EE at equal power, a methodology artifact rather than a real result.
# Mirrors how the Delta-eps=0 rate-only baseline was trained at p35 (= EE^0
# operating power). After this finishes, point RM_CHECKPOINTS['aod0.05'] in
# plotting_scenario.py at this checkpoint and re-run the sweep.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS

export EE_REWARD_MODE=sum_rate_only
export EE_TRAIN_ERROR_BOUND=0.05
export EE_POWER_BUDGET_WATT=40

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
echo "$(date): Finished SAC rate-only aod0.05 p40 nadir training!"
