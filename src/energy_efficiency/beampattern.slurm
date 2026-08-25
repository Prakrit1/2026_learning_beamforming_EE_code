#!/bin/bash
#SBATCH -J EE_beam_3gpp_aod0
#SBATCH -o EE_beam_3gpp_aod0_%j.out
#SBATCH -e EE_beam_3gpp_aod0_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-01:00:00
#SBATCH --gres=gpu:1
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --exclude=funchal,monaco,montevideo,funkia
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Beampattern generation + beamwidth/coverage quantification for the new
# aod=0.0 3GPP nadir checkpoint (job 156358, plain lwin5000 lambda mode)
# vs. the existing aod=0.05 3GPP nadir checkpoint (job 153655) and MMSE,
# all from one generate_beampatterns() call so every curve shares the same
# channel realizations. Checks whether the matched-power-MMSE rate parity
# found by my_evaluation.py / rate_power_3gpp_pilots.py (2026-08-04) is
# corroborated by healthy aiming/coverage, or whether it's a "same rate by
# luck" case the way the detlam checkpoints turned out to be. See
# beampattern.py docstring.
# Expected training_name for the new checkpoint:
# EE_dinkelbach_adaptive_lwin5000_N16K3_satg30_p75_eta0.6_rawpow
#
# No --realization arg below: beampattern.py now auto-picks the realization
# with the best worst-case coverage across MMSE + all three SAC checkpoints
# instead of a hardcoded index -- see find_best_realization() in that
# script. Pass --realization N here to force a specific one instead.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

python3 -m py_compile src/energy_efficiency/beampattern.py

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/beampattern.py
srun echo $HOSTNAME
wait

echo "$(date): Finished 3GPP nadir aod0.0 beampattern comparison!"
