#!/bin/bash
#SBATCH -J EE_beam_tanneal
#SBATCH -o EE_beam_tanneal_%j.out
#SBATCH -e EE_beam_tanneal_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-01:00:00
#SBATCH --gres=gpu:1
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --exclude=funchal,monaco,montevideo,funkia
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Generates + plots beampatterns for ALL SEVEN precoders on one realization:
# MMSE, pre-fix (rollout-lambda) SAC nadir, the four detlam checkpoints, and
# the entropy-anneal checkpoint (job 155683, 2026-07-31) -- all from one
# generate_beampatterns() call so every curve shares the same channel
# realizations. Also prints beamwidth/coverage/offset stats and diffs
# against the existing detlam-set stats if present. See
# beampattern_tanneal.py docstring.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

python3 -m py_compile src/energy_efficiency/beampattern_tanneal.py

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/beampattern_tanneal.py --realization 0
srun echo $HOSTNAME
wait

echo "$(date): Finished tanneal beampattern comparison!"
