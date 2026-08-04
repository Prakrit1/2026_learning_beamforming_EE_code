#!/bin/bash
#SBATCH -J EE_smoke_tent_anneal
#SBATCH -o EE_smoke_tent_anneal_%j.out
#SBATCH -e EE_smoke_tent_anneal_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 0-00:15:00
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Smoke test only -- 4 tiny episodes, verifies target_entropy annealing
# (soft_actor_critic.py's target_entropy tf.Variable + set_target_entropy(),
# EE_sac.py's per-episode schedule) actually takes effect inside the
# @tf.function-traced train_graph, before committing to a real multi-hour
# retrain. See smoke_test_target_entropy_anneal.py's docstring for what
# specifically this checks and why.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

python3 -m py_compile src/models/algorithms/soft_actor_critic.py src/models/EE_sac.py || exit 1

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/smoke_test_target_entropy_anneal.py
srun echo $HOSTNAME
wait

echo "$(date): Finished target_entropy annealing smoke test!"
