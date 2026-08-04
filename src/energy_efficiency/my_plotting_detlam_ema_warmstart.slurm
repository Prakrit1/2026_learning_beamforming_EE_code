#!/bin/bash
#SBATCH -J EE_plot_detlam_emaws
#SBATCH -o EE_plot_detlam_emaws_%j.out
#SBATCH -e EE_plot_detlam_emaws_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 0-00:30:00
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --exclude=funchal,monaco,montevideo,funkia
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Runs my_plotting_detlam_ema_warmstart.py: MMSE, pre-fix, pre-ema detlam
# (4), and ema0.3+warmstart detlam (4) on one plot. Needs
# my_evaluation_detlam_ema_warmstart.py's gzips -- submit with
# --dependency=afterok:<eval job>. No GPU needed.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

python3 -m py_compile src/energy_efficiency/my_plotting_detlam_ema_warmstart.py

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/my_plotting_detlam_ema_warmstart.py
srun echo $HOSTNAME
wait

echo "$(date): Finished ema+warmstart detlam plotting!"
