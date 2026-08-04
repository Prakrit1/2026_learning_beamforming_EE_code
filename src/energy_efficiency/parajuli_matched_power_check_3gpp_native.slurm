#!/bin/bash
#SBATCH -J parajuli_3gpp_native
#SBATCH -o parajuli_3gpp_native_%j.out
#SBATCH -e parajuli_3gpp_native_%j.err
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

# User-requested (2026-08-03): redo parajuli.pdf using checkpoints actually
# TRAINED at 30 dBi / 75 W (detlam mode, both error bounds -- user confirmed
# detlam-for-both over mixing detlam(0.0)/lwin5000(0.05), since no
# lwin5000-mode checkpoint exists for error=0.0 under 3GPP params). See
# parajuli_matched_power_check_3gpp_native.py docstring.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

python3 -m py_compile src/energy_efficiency/parajuli_matched_power_check_3gpp_native.py || exit 1

HOSTNAME=$(hostname)
srun nvidia-smi
srun echo $HOSTNAME
srun python3 src/energy_efficiency/parajuli_matched_power_check_3gpp_native.py
srun echo $HOSTNAME
wait

echo "$(date): Finished parajuli 3GPP native-checkpoint matched-power check!"
