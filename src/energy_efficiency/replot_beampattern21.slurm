#!/bin/bash
#SBATCH -J EE_replot_bp21
#SBATCH -o EE_replot_bp21_%j.out
#SBATCH -e EE_replot_bp21_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 0-00:15:00
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Cheap replot from the already-saved beam_patterns.gzip (--plot-only) --
# no Monte Carlo/GPU work, but per project rule everything goes through
# sbatch regardless.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2025_learning_beamforming_rsma_code
export PYTHONPATH=/home/parajuli/repos/2025_learning_beamforming_rsma_code

srun python3 src/energy_efficiency/beampattern_ee_vs_regular.py --plot-only --realization 21
wait

echo "$(date): Finished replot!"
