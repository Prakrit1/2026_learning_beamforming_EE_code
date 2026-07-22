#!/bin/bash
#SBATCH -J EE_beampattern_xerr_mmse
#SBATCH -o EE_beampattern_xerr_mmse_%j.out
#SBATCH -e EE_beampattern_xerr_mmse_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 0-00:15:00
#SBATCH --gres=gpu:1
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Replot realizations 0, 9, 13 with MMSE added to the comparison (was
# generated in the gzip all along -- generate_mmse=True -- just not
# plotted before). Cheap: no new Monte Carlo, just re-reads the saved gzip.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2025_learning_beamforming_rsma_code
export PYTHONPATH=/home/parajuli/repos/2025_learning_beamforming_rsma_code

srun python3 src/energy_efficiency/beampattern_across_error_models.py --plot-only --realization 0
srun python3 src/energy_efficiency/beampattern_across_error_models.py --plot-only --realization 9
srun python3 src/energy_efficiency/beampattern_across_error_models.py --plot-only --realization 13

echo "$(date): Finished replotting with MMSE!"
