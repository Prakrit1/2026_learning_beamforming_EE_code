#!/bin/bash
#SBATCH -J EE_rew_ee_pow
#SBATCH -o EE_rew_ee_pow_%j.out
#SBATCH -e EE_rew_ee_pow_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 0-00:15:00
#SBATCH -p wiss
#SBATCH --mem=2G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Plotting-only, no GPU/simulation. Per-episode reward and EE vs power used,
# read straight from the training log
#   outputs/metrics/EE_dinkelbach_adaptive_lwin5000_N16K3_satg30_p75_eta0.6_rawpow/base/training_error_learned_full.gzip
# (mean_reward_per_episode, episode_mean_power_dinkelbach, episode_mean_rate_dinkelbach).
# Each episode is a real (power, reward)/(power, EE) point; both are bin-
# averaged over power. NB: the Dinkelbach reward -> 0 as lambda converges
# (EE_sac.py:510), so the reward panel collapses near the operating power;
# the EE panel is the meaningful power-vs-performance curve.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/reward_ee_vs_power_training.py
srun echo $HOSTNAME
wait

echo "$(date): Finished per-episode reward/EE-vs-power figure!"
