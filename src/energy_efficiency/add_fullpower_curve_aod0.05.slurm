#!/bin/bash
#SBATCH -J EE_fullpower_curve_aod0.05
#SBATCH -o EE_fullpower_curve_aod0.05_%j.out
#SBATCH -e EE_fullpower_curve_aod0.05_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-04:00:00
#SBATCH --gres=gpu:1
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Evaluation only, no training: loads the already-trained Delta-eps=0.05 EE
# (energy_efficiency_dinkelbach_adaptive) checkpoint and re-evaluates it at
# full/total power (get_precoding_learned) instead of the clip-only
# projection, merging the result into the existing rate_power_triplet.gzip
# cache under 'sac_aod0.05_fullpower'. See add_fullpower_curve.py
# docstring -- same script as the aod=0.0/aod=0.025 curves, just parametrized.
# REQUIRES plotting_scenario.py to have already produced
# outputs/metrics/EE_lwin5000_3gpp_triplet/rate_power_triplet.gzip with the
# aod0.05 checkpoint's data (aod0.05 must be present under models/).

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/add_fullpower_curve.py aod0.05
srun echo $HOSTNAME
wait

echo "$(date): Finished EE (aod0.05) full-power inference curve!"
