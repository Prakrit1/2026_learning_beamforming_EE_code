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
# already produced and draws the 6-curve (Δε=0.00/0.025/0.05) training-error
# comparison -- for each Δε an EE^{Δε} curve (solid) and the genuine per-error
# RM^{Δε} model (dashed), both evaluated at their own energy-efficient
# (clip-only) power, so RM sits in the same power regime as its EE^{Δε}
# counterpart. See error_sweep_training_triplet.py.
# REQUIRES outputs/metrics/EE_lwin5000_3gpp_triplet/rate_power_triplet.gzip to
# already exist WITH the rm_matched_aod0.0/0.025/0.05 curves in it -- i.e. run
# plotting_scenario.slurm first with all three per-error RM checkpoints
# (RM_CHECKPOINTS in plotting_scenario.py: SAC_rateonly p35 @ Δε=0, and the
# aod0.025/aod0.05 p75 rate-only models) synced into models/.

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
