#!/bin/bash
#SBATCH -J RM_ee_power_eval
#SBATCH -o RM_ee_power_eval_%j.out
#SBATCH -e RM_ee_power_eval_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-07:00:00
#SBATCH --gres=gpu:1
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# RM (rate-only) TRAINED at the energy-efficient power and EVALUATED at that same
# power: Delta-eps=0.025 -> p38 checkpoint @ 38 W, Delta-eps=0.05 -> p40 @ 40 W,
# clip-only (native power, no 75 W downscale). Compared against EE / matched-MMSE
# read from the existing triplet gzip. NON-DESTRUCTIVE: writes its own gzip
# (outputs/metrics/EE_rateonly_ee_power/) and figure (rate_only_ee_power.*); does
# NOT touch the p75 triplet outputs.
# REQUIRES:
#   - outputs/metrics/EE_lwin5000_3gpp_triplet/rate_power_triplet.gzip (run
#     plotting_scenario.slurm first for the EE/MMSE reference curves)
#   - models/SAC_rateonly_aod0.025_N16K3_satg30_p38_eta0.6_rawpow
#   - models/SAC_rateonly_aod0.05_N16K3_satg30_p40_eta0.6_rawpow

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/rate_only_ee_power_eval.py
srun echo $HOSTNAME
wait

echo "$(date): Finished RM trained+evaluated at EE power!"
