#!/bin/bash
#SBATCH -J EE_smoke_optuna
#SBATCH -o EE_smoke_optuna_%j.out
#SBATCH -e EE_smoke_optuna_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 0-00:10:00
#SBATCH -p wiss
#SBATCH --mem=2G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Smoke test only -- no SAC training, just verifies the optuna persistent-
# storage + per-trial-exception-handling pattern before committing to three
# multi-day real searches.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2025_learning_beamforming_rsma_code
export PYTHONPATH=/home/parajuli/repos/2025_learning_beamforming_rsma_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/smoke_test_optuna_persistence.py
srun echo $HOSTNAME
wait

echo "$(date): Finished optuna persistence smoke test!"
