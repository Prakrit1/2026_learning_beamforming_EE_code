#!/bin/bash
#SBATCH -J EE_triplet_psb
#SBATCH -o EE_triplet_power_savings_bars_%j.out
#SBATCH -e EE_triplet_power_savings_bars_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-02:00:00
#SBATCH --gres=gpu:1
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Power-savings loading-bar chart (10k MC, single CSIT error point = 0.0),
# now the advisor-recommended 4-bar version: MMSE (75W budget), SAC (75W
# budget), SAC (energy-efficient, aod=0.0 checkpoint, job 156358), and MMSE
# rescaled to that SAC-EE bar's own measured power -- see
# power_savings_bars_triplet.py for the full layout. No longer needs the
# aod=0.025/0.05 checkpoints (dropped in favor of the MMSE bars).
# NOT --plot-only: this computes new MMSE samples that aren't in any
# existing gzip on disk, needs a fresh run. Pass --plot-only yourself once
# a gzip from this current 4-bar layout already exists.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/power_savings_bars_triplet.py
srun echo $HOSTNAME
wait

echo "$(date): Finished lwin5000/3GPP triplet power-savings bar chart!"
