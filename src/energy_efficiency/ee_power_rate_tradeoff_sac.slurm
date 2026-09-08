#!/bin/bash
#SBATCH -J EE_pow_rate
#SBATCH -o EE_pow_rate_%j.out
#SBATCH -e EE_pow_rate_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 0-00:15:00
#SBATCH -p wiss
#SBATCH --mem=2G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Plotting-only, no GPU/simulation: dual-axis power-vs-rate trade-off (Ha
# et al. Fig. 6 layout), transmit power on the left axis and sum rate on the
# right, one bar group for the learned policy (~35 W) and one for full-power
# RM (75 W). Both operating points come from the cached
# outputs/metrics/EE_lwin5000_3gpp_triplet/rate_power_triplet.gzip -- no new
# simulation. Run plotting_scenario.slurm first if that gzip is missing.
# Pass --error X to match a different CSIT error bound.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/ee_power_rate_tradeoff_sac.py
srun echo $HOSTNAME
wait

echo "$(date): Finished power-vs-rate trade-off figure!"
