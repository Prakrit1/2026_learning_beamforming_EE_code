#!/bin/bash
#SBATCH -J EE_3gpp_rp
#SBATCH -o EE_3gpp_rate_power_%j.out
#SBATCH -e EE_3gpp_rate_power_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-03:00:00
#SBATCH --gres=gpu:1
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Rate+power error sweep (0-0.10, 11 points, 10k MC) for the two 3GPP
# pilot checkpoints (153655 nadir / 153656 elev30) under their own system
# parameters, plus the old aod=0.05 checkpoint under the OLD system
# (20 dBi / 100 W, set inside the script via env before Config()).
# See rate_power_3gpp_pilots.py docstring.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/rate_power_3gpp_pilots.py
srun echo $HOSTNAME
wait

echo "$(date): Finished 3GPP pilots rate-power sweep!"
