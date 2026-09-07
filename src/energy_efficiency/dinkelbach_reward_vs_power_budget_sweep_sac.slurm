#!/bin/bash
#SBATCH -J EE_dinkelbach_reward_vs_budget
#SBATCH -o EE_dinkelbach_reward_vs_budget_%j.out
#SBATCH -e EE_dinkelbach_reward_vs_budget_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 0-00:15:00
#SBATCH -p wiss
#SBATCH --mem=2G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Plotting-only, no GPU/simulation: re-plots the already-cached
# ee_vs_power_budget_sweep_sac_error{X}.gzip against the actual Dinkelbach
# training reward (sumrate - lambda_ee * normalized power) instead of the
# classical rate/power ratio -- see dinkelbach_reward_vs_power_budget_sweep_sac.py's
# module docstring for why the ratio isn't the right objective to plot,
# and for the caveat that lambda_ee is an estimate (the exact training-time
# logged value no longer exists).
# REQUIRES outputs/metrics/EE_vs_transmit_power/ee_vs_power_budget_sweep_sac_error{X}.gzip
# (run ee_vs_power_budget_sweep_sac.slurm first if it doesn't exist) AND
# outputs/metrics/EE_lwin5000_3gpp_triplet/rate_power_triplet.gzip (run
# plotting_scenario.slurm first if it doesn't exist).
# Pass --error X to match a sweep run at a different CSIT error bound.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/dinkelbach_reward_vs_power_budget_sweep_sac.py
srun echo $HOSTNAME
wait

echo "$(date): Finished Dinkelbach-reward-vs-power-budget figure!"
