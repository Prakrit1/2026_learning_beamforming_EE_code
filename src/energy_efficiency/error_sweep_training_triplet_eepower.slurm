#!/bin/bash
#SBATCH -J EE_triplet_eepower
#SBATCH -o EE_error_sweep_training_triplet_eepower_%j.out
#SBATCH -e EE_error_sweep_training_triplet_eepower_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 0-00:15:00
#SBATCH -p wiss
#SBATCH --mem=2G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Plotting-only, no GPU/simulation: draws the triplet with the RM Delta-eps=0.025
# and 0.05 curves taken from the checkpoints TRAINED AND EVALUATED at the energy-
# efficient power (p38 @ 38 W, p40 @ 40 W), instead of the p75-trained RM scaled
# down. See error_sweep_training_triplet_eepower.py.
# REQUIRES both gzips to already exist:
#   - outputs/metrics/EE_lwin5000_3gpp_triplet/rate_power_triplet.gzip (plotting_scenario.slurm)
#   - outputs/metrics/EE_rateonly_ee_power/rate_only_ee_power.gzip     (rate_only_ee_power_eval.slurm)

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/error_sweep_training_triplet_eepower.py
srun echo $HOSTNAME
wait

echo "$(date): Finished error-sweep triplet (RM trained+evaluated at EE power)!"
