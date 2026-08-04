#!/bin/bash
#SBATCH -J EE_plot_tanneal
#SBATCH -o EE_plot_tanneal_%j.out
#SBATCH -e EE_plot_tanneal_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 0-00:30:00
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --exclude=funchal,monaco,montevideo,funkia
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Runs my_plotting_tanneal.py: entropy-anneal checkpoint vs MMSE, pre-fix,
# and all four detlam checkpoints. Needs my_evaluation_tanneal.py's gzip
# (detlam/pre-fix/MMSE gzips already exist) -- submit with
# --dependency=afterok:<eval job>. No GPU needed.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

python3 -m py_compile src/energy_efficiency/my_plotting_tanneal.py

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/my_plotting_tanneal.py
srun echo $HOSTNAME
wait

echo "$(date): Finished tanneal plotting!"
