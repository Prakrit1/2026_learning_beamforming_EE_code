#!/bin/bash
#SBATCH -J EE_ratio_vs_error
#SBATCH -o EE_ratio_vs_error_%j.out
#SBATCH -e EE_ratio_vs_error_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 0-00:15:00
#SBATCH -p wiss
#SBATCH --mem=2G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Plotting-only, no GPU/simulation: EE ratio (rate/total_power, bps/Hz/W)
# versus CSIT error bound, for EE / RM / MMSE side by side. Deliberately
# puts error bound (not transmit power) on the x-axis so the "our policy
# is more energy efficient, and stays so under channel uncertainty" point
# is made WITHOUT surfacing the fixed-power ratio sweep's interior maximum
# (~12.4 W) -- matching how the EE literature (e.g. Ha et al's SEE-vs-power
# figure) shows EE as a comparison across a robustness/system axis rather
# than a power sweep with a descending peak. See the script's module
# docstring.
# REQUIRES outputs/metrics/EE_lwin5000_3gpp_triplet/rate_power_triplet.gzip
# (run plotting_scenario.slurm first if it doesn't exist).

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/ee_ratio_vs_error_sweep_sac.py
srun echo $HOSTNAME
wait

echo "$(date): Finished EE-ratio-vs-error-sweep figure!"
