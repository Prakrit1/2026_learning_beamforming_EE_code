#!/bin/bash
#SBATCH -J EE_beam_detlam_emaws
#SBATCH -o EE_beam_detlam_emaws_%j.out
#SBATCH -e EE_beam_detlam_emaws_%j.err
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

# Generates + plots beampatterns for MMSE, pre-fix SAC nadir, and the four
# ema0.3+warmstart detlam retrains (2026-07-30, jobs 155291-94) on one
# realization, all from one generate_beampatterns() call. Also prints
# beamwidth/coverage/offset stats and a direct before/after comparison
# against the pre-ema detlam beampattern stats already on disk. See
# beampattern_detlam_ema_warmstart.py docstring.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

python3 -m py_compile src/energy_efficiency/beampattern_detlam_ema_warmstart.py

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/beampattern_detlam_ema_warmstart.py --realization 0
srun echo $HOSTNAME
wait

echo "$(date): Finished ema+warmstart detlam beampattern comparison!"
