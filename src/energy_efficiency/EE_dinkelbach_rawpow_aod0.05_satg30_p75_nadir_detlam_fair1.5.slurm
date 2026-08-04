#!/bin/bash
#SBATCH -J EE_3gpp_nadir_detlam_fair15
#SBATCH -o EE_3gpp_nadir_detlam_fair15_%J.out
#SBATCH -e EE_3gpp_nadir_detlam_fair15_%J.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -t 3-14:01:00
#SBATCH -p wiss
#SBATCH --mem=16G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL
#SBATCH --exclude=funchal,monaco,montevideo,funkia

# Set environment
source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS

# Deterministic-lambda retrain of the 3GPP NADIR model (2026-07-29), error
# bound 0.05 (3 of 3: aod 0.0/0.025/0.05, mirroring the old-system final
# checkpoint trio -- companion files ..._aod0.0_... and ..._aod0.025_...).
# Identical to EE_dinkelbach_rawpow_aod0.05_satg30_p75_nadir.slurm (job 153655)
# except EE_LAMBDA_DETERMINISTIC=1: the Dinkelbach lambda is now estimated from
# periodic mean-action (noise-free) eval episodes on the clipped precoder
# instead of the noise-inflated stochastic rollout. Tests the over-transmission
# mechanism from the 2026-07-28 session: the rollout-EMA lambda converged to
# ~2.09 while the deterministic policy's true rate/power ratio is ~3.8, so
# power was priced at ~half its real cost and the policy transmitted 2-3x
# above the pure-EE optimum. Expect lambda to plateau near ~3-4 and the
# deterministic power to drop toward the ~15-20 W EE-optimum if the
# hypothesis is right.
# FAIRNESS COMPANION of the non-fair detlam file: identical except
# EE_FAIRNESS_WEIGHT=1.5 (the old-system rate/fairness sweet spot).
# Expected training_name: EE_dinkelbach_adaptive_aod0.05_detlam512_N16K3_satg30_p75_eta0.6_rawpow_fair1.5
export EE_REWARD_MODE=energy_efficiency_dinkelbach_adaptive
export EE_TRAIN_ERROR_BOUND=0.05
export EE_LAMBDA_DETERMINISTIC=1
export EE_FAIRNESS_WEIGHT=1.5

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

# fail fast on a syntax error in the edited training script
python3 -m py_compile src/models/EE_sac.py || exit 1

srun nvidia-smi
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "EE_REWARD_MODE = $EE_REWARD_MODE"
echo "EE_TRAIN_ERROR_BOUND = $EE_TRAIN_ERROR_BOUND"
echo "EE_LAMBDA_DETERMINISTIC = $EE_LAMBDA_DETERMINISTIC"
echo "EE_FAIRNESS_WEIGHT = $EE_FAIRNESS_WEIGHT"

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/models/EE_sac.py

wait
echo "$(date): Finished jobs!"
