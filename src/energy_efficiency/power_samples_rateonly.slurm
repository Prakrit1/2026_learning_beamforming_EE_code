#!/bin/bash
#SBATCH -J power_samples_rateonly
#SBATCH -o power_samples_rateonly_%j.out
#SBATCH -e power_samples_rateonly_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-02:00:00
#SBATCH --gres=gpu:1
#SBATCH -p wiss
#SBATCH --mem=8G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Per-sample power/rate rollout (10k MC iterations, single config, no sweep)
# for the SAC_rateonly baseline -- see power_samples_rateonly.py docstring.
# Needs the SAC_rateonly_N16K3_satg30_p75_eta0.6_rawpow checkpoint (from
# SAC_rateonly_satg30_p75_nadir.slurm) to already be trained and saved
# under trained_models_path.

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/energy_efficiency/power_samples_rateonly.py
srun echo $HOSTNAME
wait

echo "$(date): Finished rate-only power/rate sample rollout!"
