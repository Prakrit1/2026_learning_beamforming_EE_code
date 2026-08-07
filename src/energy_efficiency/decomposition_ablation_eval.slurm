#!/bin/bash
#SBATCH -J EE_decomp_ablation_eval
#SBATCH -o EE_decomposition_ablation_eval_%j.out
#SBATCH -e EE_decomposition_ablation_eval_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-06:00:00
#SBATCH --gres=gpu:1
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Rate+power error sweep (0-0.10, 11 points, 10k MC) for the input/output
# decomposition ablation: baseline (job 1175, rad_phase-in/real_imag-out)
# vs. csi_real_imag (real_imag-in) vs. action_rad_phase (rad_phase-out),
# plus a shared full-budget MMSE curve and per-checkpoint matched-power
# MMSE curves. See decomposition_ablation_eval.py docstring. Follow up
# with my_plotting_decomposition_ablation.slurm
# (submit with --dependency=afterok:<this job>).

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/decomposition_ablation_eval.py
srun echo $HOSTNAME
wait

echo "$(date): Finished decomposition ablation rate-power sweep!"
