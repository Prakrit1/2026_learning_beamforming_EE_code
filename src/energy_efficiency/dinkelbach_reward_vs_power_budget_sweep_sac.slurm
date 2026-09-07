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
# classical rate/power ratio, using the REAL trained lambda_ee (read from
# dinkelbach_lambda_ee.txt next to the checkpoint, now that EE_sac.py
# saves it) rather than an estimate. Also plots overall transmit power
# (P_total, including circuit power/PA efficiency) on a second axis.
# REQUIRES outputs/metrics/EE_vs_transmit_power/ee_vs_power_budget_sweep_sac_error{X}.gzip
# generated from the SAME checkpoint this reads lambda_ee from -- if you
# just retrained, re-run ee_vs_power_budget_sweep_sac.slurm again FIRST so
# both come from the current (session-aware best) checkpoint, or this
# script will warn and use mismatched data.
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
