#!/bin/bash
#SBATCH -J EE_plot
#SBATCH -o EE_plot_%j.out
#SBATCH -e EE_plot_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-02:00:00
#SBATCH --gres=gpu:1
#SBATCH -p wiss
#SBATCH --mem=2scancel 38346G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Set environment
source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

# Set project root
cd /home/parajuli/repos/2025_learning_beamforming_rsma_code
export PYTHONPATH=/home/parajuli/repos/2025_learning_beamforming_rsma_code

# Execute
HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/my_plotting.py
srun echo $HOSTNAME
wait

echo "$(date): Finished evaluation!"
