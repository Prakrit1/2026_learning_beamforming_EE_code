#!/bin/bash
#SBATCH -J EE_vs_Ptx_real
#SBATCH -o EE_vs_Ptx_real_%j.out
#SBATCH -e EE_vs_Ptx_real_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 0-00:15:00
#SBATCH -p wiss
#SBATCH --mem=2G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Plotting-only, no GPU/simulation. EE vs transmit power from the REAL training
# logs (not a forced-power sweep): each episode's actually-used transmit power
# and achieved rate give a genuine (P_tx, EE) point, bin-averaged over power.
# Reads outputs/metrics/EE_dinkelbach_adaptive_lwin5000_N16K3_satg30_p75_eta0.6_rawpow/base/training_error_learned_full.gzip

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/ee_vs_transmit_power_from_training.py
srun echo $HOSTNAME
wait

echo "$(date): Finished EE-vs-transmit-power (real training data) figure!"
