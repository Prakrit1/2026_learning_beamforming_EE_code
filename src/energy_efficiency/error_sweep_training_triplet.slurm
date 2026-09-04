#!/bin/bash
#SBATCH -J EE_error_sweep_triplet
#SBATCH -o EE_error_sweep_training_triplet_%j.out
#SBATCH -e EE_error_sweep_training_triplet_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 0-00:15:00
#SBATCH -p wiss
#SBATCH --mem=2G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Plotting-only, no GPU/simulation: reads the gzip plotting_scenario.py
# already produced and draws the 6-curve (Δε=0.00/0.025/0.05, each at its
# own EE power and re-evaluated at full-power inference) training-error
# comparison with power annotations -- see error_sweep_training_triplet.py.
# REQUIRES outputs/metrics/EE_lwin5000_3gpp_triplet/rate_power_triplet.gzip
# to already exist (run plotting_scenario.slurm first if it doesn't) AND
# all three 'sac_{aod_key}_fullpower' curves merged in (run
# add_fullpower_curve.slurm, add_fullpower_curve_aod0.025.slurm, and
# add_fullpower_curve_aod0.05.slurm first if they aren't there yet).

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/error_sweep_training_triplet.py
srun echo $HOSTNAME
wait

echo "$(date): Finished error-sweep training-triplet figure!"
