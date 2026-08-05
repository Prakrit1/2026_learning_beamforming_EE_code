#!/bin/bash
#SBATCH -J EE_beam_triplet
#SBATCH -o EE_beam_triplet_%j.out
#SBATCH -e EE_beam_triplet_%j.err
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

# Beampattern generation + beamwidth/coverage quantification for the full
# lwin5000/3GPP nadir triplet (aod=0.0 job 156358, aod=0.025, aod=0.05 job
# 153655) vs. full-power MMSE, all from one generate_beampatterns() call so
# every curve shares the same channel realizations. See
# beampattern_triplet.py docstring.
# Expected training_names:
#   EE_dinkelbach_adaptive_lwin5000_N16K3_satg30_p75_eta0.6_rawpow
#   EE_dinkelbach_adaptive_aod0.025_lwin5000_N16K3_satg30_p75_eta0.6_rawpow
#   EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_satg30_p75_eta0.6_rawpow

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

python3 -m py_compile src/energy_efficiency/beampattern_triplet.py

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/beampattern_triplet.py --realization 0
srun echo $HOSTNAME
wait

echo "$(date): Finished lwin5000/3GPP triplet beampattern comparison!"
