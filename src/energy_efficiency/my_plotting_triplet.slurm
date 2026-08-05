#!/bin/bash
#SBATCH -J EE_plot_triplet
#SBATCH -o EE_plot_triplet_%j.out
#SBATCH -e EE_plot_triplet_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 0-00:30:00
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Runs my_plotting_triplet.py against the gzip produced by
# rate_power_triplet.py -- submit with --dependency=afterok:<rate_power_triplet job>.
# No simulation, plotting only -- no GPU needed.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/my_plotting_triplet.py
srun echo $HOSTNAME
wait

echo "$(date): Finished lwin5000/3GPP triplet plotting!"
