#!/bin/bash
#SBATCH -J EE_3gpp_txpow
#SBATCH -o EE_3gpp_txpower_%j.out
#SBATCH -e EE_3gpp_txpower_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-02:00:00
#SBATCH --gres=gpu:1
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Transmit-power distribution (histogram/boxplot/per-user + savings bars)
# for the two 3GPP Set-1 pilot checkpoints against the 75 W budget.
# See tx_power_3gpp_pilots.py docstring.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/tx_power_3gpp_pilots.py
srun echo $HOSTNAME
wait

echo "$(date): Finished 3GPP pilots tx power distribution!"
