#!/bin/bash
#SBATCH -J extract_ha
#SBATCH -o extract_ha_%j.out
#SBATCH -e extract_ha_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 0-00:10:00
#SBATCH -p wiss
#SBATCH --mem=2G
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

cd /home/parajuli/repos/2025_learning_beamforming_rsma_code

srun pdftotext -layout "src/energy_efficiency/Energy-Efficient_Precoding_and_Feeder-Link-Beam_Matching_Design_for_Bent-Pipe_SATCOM_Systems.pdf" "outputs/ha_extracted_text.txt"

echo "$(date): Finished Ha et al. PDF text extraction!"
