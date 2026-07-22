#!/bin/bash
#SBATCH -J extract_6gntn
#SBATCH -o extract_6gntn_%j.out
#SBATCH -e extract_6gntn_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 0-00:10:00
#SBATCH -p wiss
#SBATCH --mem=2G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

cd /home/parajuli/repos/2025_learning_beamforming_rsma_code

srun pdftotext -layout "src/energy_efficiency/6G-NTN_D4.1_r1_v00.pdf" "outputs/6gntn_d4.1_extracted_text.txt"

echo "$(date): Finished 6G-NTN PDF text extraction!"
