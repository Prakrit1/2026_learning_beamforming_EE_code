#!/bin/bash
#SBATCH -J EE_triplet_rp
#SBATCH -o EE_triplet_rate_power_%j.out
#SBATCH -e EE_triplet_rate_power_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-07:00:00
#SBATCH --gres=gpu:1
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Rate+power error sweep (0-0.10, 11 points, 10k MC) for the lwin5000/3GPP
# nadir triplet (aod=0.0 job 156358, aod=0.025, aod=0.05 job 153655): full-
# budget MMSE, SAC(err 0.0) at full power, SAC(err 0.0/0.025/0.05) energy-
# efficient (clip-only) with a matched-power MMSE curve for err 0.0 -- six
# curves total, see plotting_scenario.py's docstring for the full layout.
# REQUIRES the aod=0.025 checkpoint
# (EE_dinkelbach_adaptive_aod0.025_lwin5000_N16K3_satg30_p75_eta0.6_rawpow)
# to be present under models/ -- copy it in from wherever it was trained if
# it is missing, or the run will fail with FileNotFoundError partway through.
# Pass --plot-only (edit the srun line below) to skip the simulation and
# just re-plot an already-saved gzip.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/plotting_scenario.py --plot-only
srun echo $HOSTNAME
wait

echo "$(date): Finished lwin5000/3GPP triplet rate-power sweep!"
