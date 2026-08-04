#!/bin/bash
#SBATCH -J EE_eval_detlam_emaws
#SBATCH -o EE_eval_detlam_emaws_%j.out
#SBATCH -e EE_eval_detlam_emaws_%j.err
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

# Runs my_evaluation_detlam_ema_warmstart.py: standard rate-vs-error sweep
# for the four ema0.3+warmstart detlam retrains (2026-07-30, jobs 155291-94).
# MMSE reference is reused from the existing 3GPP nadir gzip, not re-run.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

python3 -m py_compile src/energy_efficiency/my_evaluation_detlam_ema_warmstart.py

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/my_evaluation_detlam_ema_warmstart.py
srun echo $HOSTNAME
wait

echo "$(date): Finished ema+warmstart detlam checkpoint evaluation!"
