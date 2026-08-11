#!/bin/bash
#SBATCH -J EE_plot_ratio_reward
#SBATCH -o EE_plot_ratio_reward_ablation_%j.out
#SBATCH -e EE_plot_ratio_reward_ablation_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 0-00:30:00
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Runs my_plotting_ratio_reward_ablation.py against the gzip produced by
# ratio_reward_ablation_eval.py -- submit with
# --dependency=afterok:<ratio_reward_ablation_eval job>.
# No simulation, plotting only -- no GPU needed.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/my_plotting_ratio_reward_ablation.py
srun echo $HOSTNAME
wait

echo "$(date): Finished raw-ratio-reward ablation plotting!"
