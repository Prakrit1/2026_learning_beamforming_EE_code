#!/bin/bash
#SBATCH -J EE_fairness_eval
#SBATCH -o EE_fairness_eval_%j.out
#SBATCH -e EE_fairness_eval_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-01:15:00
#SBATCH --gres=gpu:1
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Runs fairness_evaluation.py: measures rate, power, AND Jain's fairness
# index TOGETHER from the same Monte Carlo samples, across the CSIT error
# sweep, comparing the aod=0.05 baseline checkpoint (no fairness term)
# against the aod=0.05 + fairness=0.5 checkpoint (job 152699, first of the
# three trial weights to finish). Tests whether the added fairness reward
# term actually raises Jain's index, and at what rate/power cost.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/fairness_evaluation.py
srun echo $HOSTNAME
wait

echo "$(date): Finished fairness evaluation!"
