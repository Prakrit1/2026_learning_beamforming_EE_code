#!/bin/bash
#SBATCH -J EE_ratio_reward_eval
#SBATCH -o EE_ratio_reward_ablation_eval_%j.out
#SBATCH -e EE_ratio_reward_ablation_eval_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-10:00:00
#SBATCH --gres=gpu:1
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Rate+power error sweep (0-0.10, 11 points, 10k MC) for the raw-ratio-reward
# ablation (EE_ratio_reward_aod{0.0,0.035,0.05}_satg30_p75_nadir.slurm)
# against the matched-error-bound Dinkelbach-adaptive-lambda checkpoints
# (aod=0.0, aod=0.05 -- aod=0.035 has no lambda counterpart, evaluated
# standalone), plus a shared full-budget MMSE curve and per-ratio-reward-
# checkpoint matched-power MMSE curves. See ratio_reward_ablation_eval.py
# docstring. Follow up with my_plotting_ratio_reward_ablation.slurm
# (submit with --dependency=afterok:<this job>).

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/ratio_reward_ablation_eval.py
srun echo $HOSTNAME
wait

echo "$(date): Finished raw-ratio-reward ablation rate-power sweep!"
