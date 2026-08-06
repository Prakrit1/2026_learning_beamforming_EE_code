#!/bin/bash
#SBATCH -J EE_3gpp_nadir_aod0_realimaginput
#SBATCH -o EE_3gpp_nadir_aod0_realimaginput_%j.out
#SBATCH -e EE_3gpp_nadir_aod0_realimaginput_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -t 3-14:01:00
#SBATCH -p wiss
#SBATCH --mem=16G
#SBATCH --exclude=funchal,monaco,montevideo,funkia
#SBATCH --mail-user=parajuli@ant.uni-bremen.de
#SBATCH --mail-type=ALL

# Input-decomposition ablation: identical setup to
# EE_dinkelbach_rawpow_aod0.0_satg30_p75_nadir_lambdainit.slurm (job 1175,
# the current working checkpoint, reused as this ablation's baseline for
# the "rad_phase in / real_imag out" corner) except EE_CSI_FORMAT is set to
# 'real_imag' instead of the default 'rad_phase', holding the action
# (output) decomposition fixed at 'real_imag'. Tests one half of
# 04_approach.tex's "In testing, we find that magnitude-phase decomposition
# works best for the network input and real-imaginary composition works
# best for the output, though this is still under investigation" --
# specifically, whether the input-side claim holds.
#
# EE_CSI_FORMAT plumbing added this session (see config_sac_learner.py's
# get_state_args / EE_sac.py's __main__): applies
# cfg.config_learner.get_state_args['csi_format'] = 'real_imag'.
#
# Expected training_name:
# EE_dinkelbach_adaptive_lwin5000_lambdainit12.2-75_N16K3_satg30_p75_eta0.6_rawpow_csireal_imag

source /home/parajuli/miniconda3/etc/profile.d/conda.sh
conda activate gpu_beamforming

export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS

export EE_REWARD_MODE=energy_efficiency_dinkelbach_adaptive
export EE_TRAIN_ERROR_BOUND=0.0
export EE_DINKELBACH_LAMBDA_WINDOW_STEPS=5000
export EE_DINKELBACH_LAMBDA_INIT_RATE=12.2
export EE_DINKELBACH_LAMBDA_INIT_POWER=75
export EE_CSI_FORMAT=real_imag

cd /home/parajuli/repos/2026_learning_beamforming_EE_code
export PYTHONPATH=/home/parajuli/repos/2026_learning_beamforming_EE_code

python3 -m py_compile src/models/algorithms/soft_actor_critic.py src/models/EE_sac.py || exit 1

srun nvidia-smi
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "EE_REWARD_MODE = $EE_REWARD_MODE"
echo "EE_TRAIN_ERROR_BOUND = $EE_TRAIN_ERROR_BOUND"
echo "EE_DINKELBACH_LAMBDA_WINDOW_STEPS = $EE_DINKELBACH_LAMBDA_WINDOW_STEPS"
echo "EE_DINKELBACH_LAMBDA_INIT_RATE = $EE_DINKELBACH_LAMBDA_INIT_RATE"
echo "EE_DINKELBACH_LAMBDA_INIT_POWER = $EE_DINKELBACH_LAMBDA_INIT_POWER"
echo "EE_CSI_FORMAT = $EE_CSI_FORMAT"

HOSTNAME=$(hostname)
srun echo $HOSTNAME
srun python3 src/models/EE_sac.py
srun echo $HOSTNAME

wait
echo "$(date): Finished aod=0.0 3GPP nadir real_imag-input ablation!"
