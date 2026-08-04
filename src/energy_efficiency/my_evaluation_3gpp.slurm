#!/bin/bash
#SBATCH -J EE_eval_3gpp
#SBATCH -o EE_eval_3gpp_%j.out
#SBATCH -e EE_eval_3gpp_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-04:00:00
#SBATCH --gres=gpu:1
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Runs my_evaluation.py: standard-format rate-vs-error sweeps for the two
# 3GPP Set-1 models (nadir / elev30) plus MMSE in each geometry, all at
# the current 30 dBi / 75 W defaults. Old-checkpoint results on disk are
# NOT touched (see my_evaluation.py's header comment).

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/my_evaluation.py
srun echo $HOSTNAME
wait

echo "$(date): Finished 3GPP model evaluation!"
