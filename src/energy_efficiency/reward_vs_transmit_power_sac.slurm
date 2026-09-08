#!/bin/bash
#SBATCH -J EE_reward_P
#SBATCH -o EE_reward_P_%j.out
#SBATCH -e EE_reward_P_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 0-00:15:00
#SBATCH -p wiss
#SBATCH --mem=2G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Plotting-only, no GPU/simulation: the training reward
#   reward(P) = R(P) - lambda_ee * P_total(P) / P_max
# vs transmit power. Concave rate minus a linear price -> a maximum at the
# policy's operating point (~35 W), in contrast to the EE optimum (~12.4 W).
# Reads the cached rate(P) sweep
#   outputs/metrics/EE_vs_transmit_power/ee_vs_power_sweep_sac_error0.gzip
# (run ee_vs_transmit_power_sweep_sac.py first if missing) and lambda_ee from
# the checkpoint. If the checkpoint has no dinkelbach_lambda_ee.txt, pass
# --lambda X (your trained value, ~3.47). Pass --error X for a different bound.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/reward_vs_transmit_power_sac.py
srun echo $HOSTNAME
wait

echo "$(date): Finished reward-vs-transmit-power figure!"
