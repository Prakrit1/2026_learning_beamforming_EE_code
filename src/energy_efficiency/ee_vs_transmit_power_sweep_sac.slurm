#!/bin/bash
#SBATCH -J EE_vs_Ptx
#SBATCH -o EE_vs_Ptx_%j.out
#SBATCH -e EE_vs_Ptx_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 0-00:40:00
#SBATCH -p wiss
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Energy-efficiency-vs-transmit-power figure for the deployed EE policy
# ('aod0.0' Dinkelbach checkpoint): left axis EE(P), right (twin) axis the same
# policy's sum rate(P), both over the constant-power sweep, with the ~35 W
# operating point marked on both curves. See the module docstring.
#
# This launcher RECOMPUTES the sweep (~400k network forward passes), hence the
# GPU request. It reads the deployed operating point from
# outputs/metrics/EE_lwin5000_3gpp_triplet/rate_power_triplet.gzip.
# Once the cache exists, add --plot-only to the python line below (and drop the
# --gres/mem back down) for a fast, no-GPU replot.
# Pass --error X to match a different CSIT error bound.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/ee_vs_transmit_power_sweep_sac.py
srun echo $HOSTNAME
wait

echo "$(date): Finished EE-vs-transmit-power figure!"
