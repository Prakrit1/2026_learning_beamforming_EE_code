#!/bin/bash
#SBATCH -J EE_vs_power_sac_paper
#SBATCH -o EE_vs_transmit_power_sac_paper_%j.out
#SBATCH -e EE_vs_transmit_power_sac_paper_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-04:00:00
#SBATCH --gres=gpu:1
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Same sweep as ee_vs_transmit_power_sweep_sac.slurm (SAC aod=0.0 checkpoint,
# raw-direction EE vs. fixed transmit power, 1-75W, 40 points, 10k MC per
# point, CSIT error bound 0.0, overlaid against the saved MMSE curve), but
# with --hide-maximizer so the paper-ready figure omits the per-direction
# EE-maximizer (12.4 W) marker/legend entry -- see ee_vs_transmit_power_sweep_sac.py
# docstring and its HIDE_MAXIMIZER flag for why: showing that number next to
# the deployed policy's actual, higher operating power (~35-40W) elsewhere
# in the paper reads as a contradiction, when it's really two different
# things (a per-direction theoretical ceiling vs. the policy's own learned
# choice). The 12.4 W value is still printed to the .out log, just not
# drawn on the figure.
# REQUIRES ee_vs_transmit_power_sweep.slurm to have run first (or an
# existing outputs/metrics/EE_vs_transmit_power/ee_vs_power_sweep_error{X}.gzip)
# for the MMSE overlay -- runs fine without it too, just skips that curve.
# Output: reports/figures/{pdf,jpg,png}/ee_vs_transmit_power_sac_error0_paper.*
# (the _paper suffix keeps this from overwriting the existing diagnostic
# figure produced by ee_vs_transmit_power_sweep_sac.slurm).

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/ee_vs_transmit_power_sweep_sac.py --hide-maximizer
srun echo $HOSTNAME
wait

echo "$(date): Finished SAC EE-vs-transmit-power sweep (paper figure, no maximizer marker)!"
