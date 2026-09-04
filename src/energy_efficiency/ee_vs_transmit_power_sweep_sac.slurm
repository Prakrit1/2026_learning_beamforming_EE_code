#!/bin/bash
#SBATCH -J EE_vs_power_sac
#SBATCH -o EE_vs_transmit_power_sac_%j.out
#SBATCH -e EE_vs_transmit_power_sac_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-04:00:00
#SBATCH --gres=gpu:1
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# SAC (aod=0.0 checkpoint, job 156358, same one plotting_scenario.py and
# power_savings_bars_triplet.py use) raw-direction EE vs. fixed transmit
# power sweep (1-75W, 40 points, 10k MC per point) at CSIT error bound 0.0.
# Plots BOTH the 12.4W fixed-power EE optimum (this sweep's own result) and
# the deployed policy's actual ~35W adaptive-power operating point (a gold
# star, pulled from rate_power_triplet.gzip) on the same axes with short,
# distinct labels, so the two numbers read as answers to different
# questions rather than a contradiction -- see
# ee_vs_transmit_power_sweep_sac.py's comments for the full reasoning.
# REQUIRES ee_vs_transmit_power_sweep.slurm to have run first (or an
# existing outputs/metrics/EE_vs_transmit_power/ee_vs_power_sweep_error{X}.gzip)
# for the MMSE overlay -- runs fine without it too, just skips that curve.
# Also reads outputs/metrics/EE_lwin5000_3gpp_triplet/rate_power_triplet.gzip
# for the deployed-policy star -- runs fine without it too, just skips that
# marker (requires plotting_scenario.slurm to have run first).
# Pass --error X to sweep at a different CSIT error bound, or --plot-only
# to re-plot from an existing gzip without re-running the 10k-iteration MC
# sweep (edit the srun line below for either).

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/ee_vs_transmit_power_sweep_sac.py
srun echo $HOSTNAME
wait

echo "$(date): Finished SAC EE-vs-transmit-power sweep!"
