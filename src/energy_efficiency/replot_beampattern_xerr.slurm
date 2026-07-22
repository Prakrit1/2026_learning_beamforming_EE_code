#!/bin/bash
#SBATCH -J EE_beampattern_xerr_replot
#SBATCH -o EE_beampattern_xerr_replot_%j.out
#SBATCH -e EE_beampattern_xerr_replot_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 0-00:15:00
#SBATCH --gres=gpu:1
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Cheap replot from the already-generated beam_patterns.gzip (job 152483) --
# no new Monte Carlo needed. Realization 9: dramatic divergence between the
# three error-trained EE checkpoints (sum rate 0.58/1.63/4.09 for
# aod0.0/aod0.025/aod0.05). Realization 13: all three closely agree
# (4.69/4.34/3.80, all near MMSE's 4.64).

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2025_learning_beamforming_rsma_code
export PYTHONPATH=/home/parajuli/repos/2025_learning_beamforming_rsma_code

srun python3 src/energy_efficiency/beampattern_across_error_models.py --plot-only --realization 9
srun python3 src/energy_efficiency/beampattern_across_error_models.py --plot-only --realization 13

echo "$(date): Finished replotting realizations 9 and 13!"
