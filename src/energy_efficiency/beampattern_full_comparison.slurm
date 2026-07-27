#!/bin/bash
#SBATCH -J EE_beam_full
#SBATCH -o EE_beam_full_%j.out
#SBATCH -e EE_beam_full_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-01:00:00
#SBATCH --gres=gpu:1
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Generates + plots beampatterns for ALL SIX precoders on one realization:
# MMSE, regular (100W) SAC, EE aod=0.05 baseline, and EE aod=0.05 fairness
# 0.5/1.5/3.0 -- all from one generate_beampatterns() call so every curve
# shares the same channel realizations. Also prints dropped-user candidate
# realizations. See beampattern_full_comparison.py docstring.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/beampattern_full_comparison.py --realization 0
srun echo $HOSTNAME
wait

echo "$(date): Finished full beampattern comparison!"
