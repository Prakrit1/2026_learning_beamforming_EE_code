#!/bin/bash
#SBATCH -J EE_triplet_psb
#SBATCH -o EE_triplet_power_savings_bars_%j.out
#SBATCH -e EE_triplet_power_savings_bars_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-02:00:00
#SBATCH --gres=gpu:1
#SBATCH -p wiss
#SBATCH --mem=4G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Power-savings loading-bar chart (10k MC, single CSIT error point = 0.0) for
# the lwin5000/3GPP nadir triplet (aod=0.0 job 156358, aod=0.025, aod=0.05
# job 153655) plus the aod=0.0 checkpoint's full-power reference bar -- see
# power_savings_bars_triplet.py's docstring for the full layout. Only one
# error point per checkpoint (vs. plotting_scenario.slurm's 11), so this is
# much cheaper -- 2h budget is generous headroom, not a measured requirement.
# REQUIRES the aod=0.025 checkpoint
# (EE_dinkelbach_adaptive_aod0.025_lwin5000_N16K3_satg30_p75_eta0.6_rawpow)
# to be present under models/.
# Pass --plot-only (edit the srun line below) to skip the simulation and
# just re-plot an already-saved gzip.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/power_savings_bars_triplet.py --plot-only
srun echo $HOSTNAME
wait

echo "$(date): Finished lwin5000/3GPP triplet power-savings bar chart!"
