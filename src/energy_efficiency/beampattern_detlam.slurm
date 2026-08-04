#!/bin/bash
#SBATCH -J EE_beam_detlam
#SBATCH -o EE_beam_detlam_%j.out
#SBATCH -e EE_beam_detlam_%j.err
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

# Generates + plots beampatterns for ALL SIX precoders on one realization:
# MMSE, pre-fix (rollout-lambda) SAC nadir, and the four detlam checkpoints
# (aod0.0/aod0.05, with/without fairness=1.5) -- all from one
# generate_beampatterns() call so every curve shares the same channel
# realizations. Also prints beamwidth/coverage stats. See
# beampattern_detlam.py docstring.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

python3 -m py_compile src/energy_efficiency/beampattern_detlam.py

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/beampattern_detlam.py --realization 0
srun echo $HOSTNAME
wait

echo "$(date): Finished detlam beampattern comparison!"
