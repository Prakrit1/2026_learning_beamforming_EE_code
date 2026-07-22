#!/bin/bash
#SBATCH -J EE_beam_robust
#SBATCH -o EE_beam_robust_%j.out
#SBATCH -e EE_beam_robust_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 0-00:20:00
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Quantifies beam narrowness vs. CSIT-error robustness directly from the
# two ALREADY-SAVED beampattern gzips (EE_beampattern_comparison_N16K3_aod0.05,
# EE_beampattern_across_error_models_N16K3) -- no new Monte Carlo, no GPU,
# just re-reading pickled results and computing beamwidth / coverage-ratio
# statistics. See beampattern_robustness_analysis.py docstring.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2025_learning_beamforming_rsma_code
export PYTHONPATH=/home/parajuli/repos/2025_learning_beamforming_rsma_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/beampattern_robustness_analysis.py
srun echo $HOSTNAME
wait

echo "$(date): Finished beampattern robustness analysis!"
