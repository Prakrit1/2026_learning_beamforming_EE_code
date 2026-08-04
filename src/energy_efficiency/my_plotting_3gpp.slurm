#!/bin/bash
#SBATCH -J EE_plot_3gpp
#SBATCH -o EE_plot_3gpp_%j.out
#SBATCH -e EE_plot_3gpp_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 0-00:30:00
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Runs my_plotting.py: (1) current 3GPP models vs MMSE rate curves (from
# my_evaluation.py's gzips -- submit with --dependency=afterok:<eval job>),
# (2) the old-system benchmark figure from the existing pre-2026-07-27
# gzips on disk. No simulation, plotting only -- no GPU needed.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/my_plotting.py
srun echo $HOSTNAME
wait

echo "$(date): Finished 3GPP plotting!"
