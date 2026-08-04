#!/bin/bash
#SBATCH -J EE_eval_tanneal
#SBATCH -o EE_eval_tanneal_%j.out
#SBATCH -e EE_eval_tanneal_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-04:00:00
#SBATCH --gres=gpu:1
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --exclude=funchal,monaco,montevideo,funkia
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Runs my_evaluation_tanneal.py: standard rate-vs-error sweep for the
# target_entropy-annealing checkpoint (job 155683, 2026-07-31). MMSE
# reference is reused from the existing 3GPP nadir gzip, not re-run.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

python3 -m py_compile src/energy_efficiency/my_evaluation_tanneal.py

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/my_evaluation_tanneal.py
srun echo $HOSTNAME
wait

echo "$(date): Finished tanneal checkpoint evaluation!"
