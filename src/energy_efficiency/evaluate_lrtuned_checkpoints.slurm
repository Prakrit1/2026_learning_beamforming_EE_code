#!/bin/bash
#SBATCH -J EE_eval_lrtuned
#SBATCH -o EE_eval_lrtuned_%j.out
#SBATCH -e EE_eval_lrtuned_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-08:00:00
#SBATCH --gres=gpu:1
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Runs the full canonical my_evaluation.py (re-runs the three default-LR
# blocks too, matching this project's single-canonical-script convention),
# now with three added blocks for the tuned-LR checkpoints (jobs
# 152303/152304/152305, using each error bound's best lr_critic/lr_actor
# from the Optuna searches). Doubled block count (3 -> 6) vs.
# evaluate_error_bound_checkpoints.slurm, so time budget bumped to 8h.
# NOTE: the aod0.025 tuned-LR checkpoint's entropy_scale_alpha diverged
# during training (exploded to ~1e19) -- expect that block's numbers to
# look degenerate, see my_evaluation.py's comment on that block.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2025_learning_beamforming_rsma_code
export PYTHONPATH=/home/parajuli/repos/2025_learning_beamforming_rsma_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/my_evaluation.py
srun echo $HOSTNAME
wait

echo "$(date): Finished lr-tuned + default-LR checkpoint evaluation!"
