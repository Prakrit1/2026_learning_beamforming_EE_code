#!/bin/bash
#SBATCH -J EE_3gpp_elev30
#SBATCH -o EE_3gpp_elev30_%J.out
#SBATCH -e EE_3gpp_elev30_%J.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -t 3-14:01:00
#SBATCH -p wiss
#SBATCH --mem=16G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Set environment
source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS

# 3GPP-aligned pilot retrain #2 of 2 (2026-07-27): 30 DEG ELEVATION geometry.
# Identical to the nadir companion job
# (EE_dinkelbach_rawpow_aod0.05_satg30_p75_nadir.slurm) except
# EE_TARGET_ELEVATION_DEG=30, which config.py translates into shifting the
# user-cluster center 7.67 deg of earth-central angle off nadir (center
# 82.33 deg, slant range ~1074 km vs 600 km, +5.1 dB path loss) -- the
# TR 38.821 Table 6.1.3.2-1 (p. 52) LEO link-budget target elevation.
# Physics expectations vs. nadir: worse link budget, users off array
# broadside (smaller sin(theta) amplifies the cos-AoD CSIT error's angular
# offset), tighter user angular spread (more correlated channels).
# Expected training_name: EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_satg30_p75_elev30_eta0.6_rawpow
export EE_REWARD_MODE=energy_efficiency_dinkelbach_adaptive
export EE_TRAIN_ERROR_BOUND=0.05
export EE_DINKELBACH_LAMBDA_WINDOW_STEPS=5000
export EE_TARGET_ELEVATION_DEG=30

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

srun nvidia-smi
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "EE_REWARD_MODE = $EE_REWARD_MODE"
echo "EE_TRAIN_ERROR_BOUND = $EE_TRAIN_ERROR_BOUND"
echo "EE_DINKELBACH_LAMBDA_WINDOW_STEPS = $EE_DINKELBACH_LAMBDA_WINDOW_STEPS"
echo "EE_TARGET_ELEVATION_DEG = $EE_TARGET_ELEVATION_DEG"

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/models/EE_sac.py

wait
echo "$(date): Finished jobs!"
