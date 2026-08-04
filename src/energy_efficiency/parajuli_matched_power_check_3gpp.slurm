#!/bin/bash
#SBATCH -J parajuli_3gpp
#SBATCH -o parajuli_3gpp_%j.out
#SBATCH -e parajuli_3gpp_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-03:00:00
#SBATCH --gres=gpu:1
#SBATCH -p wiss
#SBATCH --mem=8G
#SBATCH --exclude=funchal,monaco,montevideo,funkia
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# User-requested (2026-08-03): re-evaluate the two OLD-system SAC checkpoints
# (EE_dinkelbach_adaptive_lwin5000_N16K3_eta0.6_rawpow, trained error=0.0, and
# ..._aod0.05_lwin5000_N16K3_eta0.6_rawpow, trained error=0.05) under the
# 3GPP Set-1 system parameters (30 dBi / 75 W) instead of the 20 dBi / 100 W
# they were actually trained under -- an out-of-distribution eval, not a
# native-system result. See parajuli_matched_power_check_3gpp.py docstring.
# Full rate+power sweep for both SAC checkpoints (11 error points, 10k MC
# each) plus full-budget MMSE (75W) and matched-power MMSE (matched to SAC
# err=0.0's own measured power at error=0), all under 30 dBi -- none of the
# existing on-disk gzips are reusable since they predate the 3GPP config
# change and were computed at 20 dBi.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

python3 -m py_compile src/energy_efficiency/parajuli_matched_power_check_3gpp.py || exit 1

HOSTNAME=$(hostname)
srun nvidia-smi
srun echo $HOSTNAME
srun python3 src/energy_efficiency/parajuli_matched_power_check_3gpp.py
srun echo $HOSTNAME
wait

echo "$(date): Finished parajuli 3GPP matched-power check!"
