#!/bin/bash
#SBATCH -J EE_3gpp_nadir_aod0
#SBATCH -o EE_3gpp_nadir_aod0_%j.out
#SBATCH -e EE_3gpp_nadir_aod0_%j.err
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

# aod=0.0 counterpart to EE_dinkelbach_rawpow_aod0.05_satg30_p75_nadir.slurm
# (job 153655), trained 2026-08-03 with EE_sac.py/soft_actor_critic.py
# reverted to the last-committed, non-experimental state (detlam/EMA/
# warm-start/entropy-anneal/power-conditioned additions stashed --
# `git stash list` on this branch has the snapshot if that work is wanted
# back later). Same reward/lambda setup (plain lwin5000 EMA Dinkelbach,
# matching the doc's own documented mechanism) and same 3GPP Set-1 system
# parameters (sat_gain_dBi=30, power_constraint_watt=75 -- config.py
# defaults, no override needed) as the existing aod=0.05 nadir checkpoint,
# so the two form a clean, non-mixed-lambda-mode pair for a matched-power
# SAC-vs-MMSE comparison at 30 dBi/75 W (see parajuli_matched_power_check_
# *.py -- the detlam-checkpoint version of that comparison showed a large,
# already-previously-documented SAC-vs-matched-MMSE gap; this retrain tests
# whether that gap was specific to the detlam/EMA/warmstart mechanisms).
# Expected training_name: EE_dinkelbach_adaptive_lwin5000_N16K3_satg30_p75_eta0.6_rawpow

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS

export EE_REWARD_MODE=energy_efficiency_dinkelbach_adaptive
export EE_TRAIN_ERROR_BOUND=0.0
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
echo "$(date): Finished aod=0.0 3GPP nadir retrain!"
