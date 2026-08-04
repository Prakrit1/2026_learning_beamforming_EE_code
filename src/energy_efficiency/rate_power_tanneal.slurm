#!/bin/bash
#SBATCH -J EE_tanneal_rp
#SBATCH -o EE_tanneal_rate_power_%j.out
#SBATCH -e EE_tanneal_rate_power_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-02:00:00
#SBATCH --gres=gpu:1
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --exclude=funchal,monaco,montevideo,funkia
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Rate+power error sweep (0-0.10, 11 points, 10k MC) for the entropy-anneal
# checkpoint (job 155683, 2026-07-31). MMSE + pre-fix SAC nadir curves reused
# from rate_power_3gpp_pilots.gzip, and the four detlam curves reused from
# rate_power_detlam.gzip -- none re-simulated. See rate_power_tanneal.py
# docstring.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

python3 -m py_compile src/energy_efficiency/rate_power_tanneal.py

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/rate_power_tanneal.py
srun echo $HOSTNAME
wait

echo "$(date): Finished tanneal rate-power sweep!"
