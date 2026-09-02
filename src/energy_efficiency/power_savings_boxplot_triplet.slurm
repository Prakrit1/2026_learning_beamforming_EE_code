#!/bin/bash
#SBATCH -J EE_triplet_psbox
#SBATCH -o EE_triplet_power_savings_boxplot_%j.out
#SBATCH -e EE_triplet_power_savings_boxplot_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-02:00:00
#SBATCH --gres=gpu:1
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# 3-panel power-distribution figure (overlaid histogram, total-power
# boxplot, per-user grouped boxplot) for the same 4 curves as
# power_savings_bars_triplet.py's bar chart (Figure 2): MMSE, RM (aod=0.0
# checkpoint at full-power inference), EE (aod=0.0 checkpoint, clip-only),
# MMSE rescaled to EE's own measured power. See power_savings_boxplot_
# triplet.py for the full layout and plot_power_savings_comparison in
# tx_power_distribution.py for the plotting code.
# Shares outputs/metrics/EE_lwin5000_3gpp_triplet/power_savings_bars.gzip
# with power_savings_bars_triplet.py -- if that job already ran, this one
# reuses its cached samples instead of re-running the 10k-iteration MC
# sweep. Run power_savings_bars_triplet.slurm first if you want a single
# shared MC run backing both figures.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/power_savings_boxplot_triplet.py
srun echo $HOSTNAME
wait

echo "$(date): Finished lwin5000/3GPP triplet power-savings boxplot figure!"
