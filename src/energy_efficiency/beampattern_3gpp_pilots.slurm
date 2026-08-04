#!/bin/bash
#SBATCH -J EE_3gpp_beam
#SBATCH -o EE_3gpp_beampattern_%j.out
#SBATCH -e EE_3gpp_beampattern_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-01:30:00
#SBATCH --gres=gpu:1
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Beampattern generation + beamwidth/coverage quantification for the two
# 3GPP pilot checkpoints (153655 nadir / 153656 elev30), each in its own
# geometry with a dynamically centered angle sweep (elev30 users sit far
# off broadside). Verifies handoff item 4c: learned beamwidth stays in the
# ~2-4.5 deg Set-1-consistent range. See beampattern_3gpp_pilots.py.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/beampattern_3gpp_pilots.py
srun echo $HOSTNAME
wait

echo "$(date): Finished 3GPP pilots beampattern analysis!"
