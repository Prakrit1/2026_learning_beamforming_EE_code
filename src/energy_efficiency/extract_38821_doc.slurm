#!/bin/bash
#SBATCH -J extract_38821
#SBATCH -o extract_38821_%j.out
#SBATCH -e extract_38821_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 0-00:10:00
#SBATCH -p wiss
#SBATCH --mem=2G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

cd /home/parajuli/repos/2025_learning_beamforming_rsma_code
export PYTHONPATH=/home/parajuli/repos/2025_learning_beamforming_rsma_code

srun python3 src/energy_efficiency/extract_38821_doc.py

echo "$(date): Finished 3GPP TR 38.821 text extraction!"
