#!/bin/bash
#SBATCH -J extract_ppa
#SBATCH -o extract_ppa_%j.out
#SBATCH -e extract_ppa_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 0-00:10:00
#SBATCH -p wiss
#SBATCH --mem=2G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

cd /home/parajuli/repos/2025_learning_beamforming_rsma_code

srun pdftotext -layout "src/energy_efficiency/Performance_Power_and_Area_Design_Trade-offs_in_Mi.pdf" "outputs/ppa_extracted_text.txt"

echo "$(date): Finished PPA PDF text extraction!"
