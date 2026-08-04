#!/bin/bash
#SBATCH -J EE_pcond_eval
#SBATCH -o EE_pcond_eval_%j.out
#SBATCH -e EE_pcond_eval_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-08:00:00
#SBATCH --gres=gpu:1
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --exclude=funchal,monaco,montevideo,funkia
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Offline EE-optimal-power grid search + downstream evaluation for the
# power-conditioned checkpoint (job 155697, 2026-07-31): 25-point log-spaced
# power grid (1-75 W, 3000 MC each) at CSIT error=0.05 to find P* = argmax
# EE(P), then a rate-vs-error sweep at fixed P* (11 pts, 10k MC) and a
# matched-power MMSE sweep at the same P*, compared against the detlam
# checkpoints' own operating points at error=0.05. See
# power_conditioned_evaluation.py docstring. ~3000*25 + 10000*11*2 ~= 295k
# network/precoder calls total -- budgeted generously at 8h given this is a
# new, unprofiled script (other 4-checkpoint x 11-point x 10k-MC sweeps in
# this directory finish well inside 4h).

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

python3 -m py_compile src/energy_efficiency/power_conditioned_evaluation.py

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/power_conditioned_evaluation.py
srun echo $HOSTNAME
wait

echo "$(date): Finished power-conditioned offline EE-optimal-power evaluation!"
