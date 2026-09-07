#!/bin/bash
#SBATCH -J EE_vs_power_budget_sac
#SBATCH -o EE_vs_power_budget_sac_%j.out
#SBATCH -e EE_vs_power_budget_sac_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-04:00:00
#SBATCH --gres=gpu:1
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Companion to ee_vs_transmit_power_sweep_sac.py, answering a different
# question: instead of forcing every channel to a fixed transmit power
# (that script's 12.4W-peaking sweep), this keeps the same aod=0.0
# checkpoint's ADAPTIVE clip-only projection and sweeps the available
# power BUDGET fed to that clip step from 1-75W (40 points, 10k MC per
# point, CSIT error bound 0.0). Expect rise-then-plateau, not
# rise-then-fall: clip-only projection only clips DOWN when the raw
# network output exceeds budget, it never pushes power UP toward it --
# see ee_vs_power_budget_sweep_sac.py's module docstring for the full
# reasoning this script came out of.
# REQUIRES the trained aod=0.0 checkpoint under models/ (same one
# plotting_scenario.py and ee_vs_transmit_power_sweep_sac.py use). No
# other gzip dependency -- this is a self-contained sweep.
# Pass --error X to sweep at a different CSIT error bound, or --plot-only
# to re-plot from an existing gzip without re-running the 10k-iteration MC
# sweep (edit the srun line below for either).

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/ee_vs_power_budget_sweep_sac.py
srun echo $HOSTNAME
wait

echo "$(date): Finished SAC EE-vs-power-budget sweep!"
