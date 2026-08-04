#!/bin/bash
#SBATCH -J EE_smoke_pcond
#SBATCH -o EE_smoke_pcond_%j.out
#SBATCH -e EE_smoke_pcond_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 0-00:15:00
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Smoke test only -- 4 tiny episodes, verifies the power-conditioned reward
# mode (power budget as extra actor input, always-rescale-to-that-budget
# precoder, pure-rate reward) actually runs without crashing and produces
# sane sampled-power / achieved-rate numbers, before committing to a real
# multi-hour training run. See smoke_test_power_conditioned.py's docstring.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

python3 -m py_compile src/models/EE_sac.py || exit 1

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/smoke_test_power_conditioned.py
srun echo $HOSTNAME
wait

echo "$(date): Finished power-conditioned smoke test!"
