#!/bin/bash
#SBATCH -J EE_vs_RM_ratio
#SBATCH -o EE_vs_RM_ratio_%j.out
#SBATCH -e EE_vs_RM_ratio_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 0-00:15:00
#SBATCH -p wiss
#SBATCH --mem=2G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Plotting-only, no GPU/simulation: two-bar comparison of the classical EE
# ratio (rate/total_power, bps/Hz/W) at exactly two single operating
# points -- EE's own deployed point (~35 W, where the real trained
# lambda_ee actually drives it) vs RM at full 75 W power. Both numbers
# come straight from the already-cached rate_power_triplet.gzip, no new
# simulation. See ee_vs_rm_ratio_comparison_sac.py's module docstring for
# why this sidesteps the "why isn't EE at its 12.4 W ratio optimum"
# question entirely.
# REQUIRES outputs/metrics/EE_lwin5000_3gpp_triplet/rate_power_triplet.gzip
# (run plotting_scenario.slurm first if it doesn't exist) AND a checkpoint
# with dinkelbach_lambda_ee.txt saved (post-instrumentation retrain).
# Pass --error X to match a different CSIT error bound.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/ee_vs_rm_ratio_comparison_sac.py
srun echo $HOSTNAME
wait

echo "$(date): Finished EE-vs-RM ratio comparison figure!"
