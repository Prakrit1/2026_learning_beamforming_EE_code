#!/bin/bash
#SBATCH -J EE_vs_Ptx
#SBATCH -o EE_vs_Ptx_%j.out
#SBATCH -e EE_vs_Ptx_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 0-00:15:00
#SBATCH -p wiss
#SBATCH --mem=2G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Dual-axis energy-efficiency-vs-transmit-power figure for the deployed SAC
# policy: EE(P) (left, concave) and sum rate(P) (right, saturating) over the
# constant-power sweep, with the proposed (~35 W) and full-power (75 W)
# operating points marked so the EE gain reads as an honest move along the
# rate/power curve. See ee_vs_transmit_power_sweep_sac.py's module docstring.
#
# This launcher is PLOT-ONLY (fast, no GPU): it replots from the cached
# outputs/metrics/EE_vs_transmit_power/ee_vs_power_sweep_sac_error0.gzip and
# reads the deployed operating point from
# outputs/metrics/EE_lwin5000_3gpp_triplet/rate_power_triplet.gzip.
# To RECOMPUTE the sweep from scratch (400k network forward passes, needs the
# GPU partition and a longer -t), drop the --plot-only flag below.
# Pass --error X to match a different CSIT error bound.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/ee_vs_transmit_power_sweep_sac.py --plot-only
srun echo $HOSTNAME
wait

echo "$(date): Finished EE-vs-transmit-power figure!"
