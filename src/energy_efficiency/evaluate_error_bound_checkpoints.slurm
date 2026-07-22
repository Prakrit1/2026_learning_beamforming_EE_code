#!/bin/bash
#SBATCH -J EE_eval_errbound
#SBATCH -o EE_eval_errbound_%j.out
#SBATCH -e EE_eval_errbound_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-04:00:00
#SBATCH --gres=gpu:1
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Runs the full canonical my_evaluation.py (re-runs already-done blocks
# too, matching this project's single-canonical-script convention) for the
# three raw-power-fixed Dinkelbach checkpoints trained at error=0.0, 0.025,
# and 0.05 (jobs 147505/148538/147506, "Finding 3" + the paper's own
# three-point training match). ~90-130 min expected given the block count.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2025_learning_beamforming_rsma_code
export PYTHONPATH=/home/parajuli/repos/2025_learning_beamforming_rsma_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/my_evaluation.py
srun echo $HOSTNAME
wait

echo "$(date): Finished error-bound checkpoint evaluation!"
