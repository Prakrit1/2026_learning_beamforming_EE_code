#!/bin/bash
#SBATCH -J EE_beam_full_r11
#SBATCH -o EE_beam_full_r11_%j.out
#SBATCH -e EE_beam_full_r11_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 0-00:15:00
#SBATCH --gres=gpu:1
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Plot-only rerun of beampattern_full_comparison.py for realization 11
# (the clearest "EE baseline drops a user" candidate found by the main
# job's dropped-user diagnostic) -- reuses the already-saved gzip, no new
# Monte Carlo.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/beampattern_full_comparison.py --plot-only --realization 11
srun echo $HOSTNAME
wait

echo "$(date): Finished plot-only realization 11!"
