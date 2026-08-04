#!/bin/bash
#SBATCH -J EE_pow_scale
#SBATCH -o EE_power_scaling_%j.out
#SBATCH -e EE_power_scaling_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-01:00:00
#SBATCH --gres=gpu:1
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Post-hoc power-scaling diagnostic: scales each trained checkpoint's
# clipped precoder by s in (0.1..1.0] and measures EE(s) on paired channel
# samples. Decides whether the 3GPP pilots' full-budget operating point is
# the true EE optimum (EE peaks at s=1) or a clip-boundary training
# artifact (EE peaks at s<1). Old aod=0.05 checkpoint runs as control.
# See ee_power_scaling_diagnostic.py docstring.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/ee_power_scaling_diagnostic.py
srun echo $HOSTNAME
wait

echo "$(date): Finished power scaling diagnostic!"
