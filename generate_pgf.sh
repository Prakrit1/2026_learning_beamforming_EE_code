#!/bin/bash
# Regenerate the four .pgf figures in reports/figures/pgf/ after editing the
# plotting scripts (e.g. the P -> P_eval label + bigger-legend change).
#
# Run this on a machine that HAS a latex binary on PATH (login node / your own
# terminal) -- matplotlib only writes the .pgf if it finds xelatex/lualatex/
# pdflatex; on the compute nodes there is no latex, so the .pgf is silently
# skipped there. All four scripts below are fast: three re-plot from a cached
# gzip, and plotting_scenario is run with --plot-only so it re-plots from its
# cache instead of recomputing the 10k-MC sweep. (This also writes the matching
# pdf/png/jpg alongside each .pgf.)
set -euo pipefail

# Repo root = wherever this script lives.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO"

# Python interpreter: override if your local env differs, e.g.
#   PYTHON=python ./generate_pgf.sh
#   PYTHON=.venv/bin/python ./generate_pgf.sh
PYTHON="${PYTHON:-python3}"

# --- conda env (only on the cluster; skipped automatically when run locally) ---
if [ -f /home/parajuli/miniconda3/etc/profile.d/conda.sh ]; then
    source /home/parajuli/miniconda3/etc/profile.d/conda.sh
    conda activate gpu_beamforming
fi

# --- latex check: without it, no .pgf gets written (only pdf/png/jpg) ---
if ! command -v xelatex >/dev/null 2>&1 \
   && ! command -v lualatex >/dev/null 2>&1 \
   && ! command -v pdflatex >/dev/null 2>&1; then
    echo "WARNING: no xelatex/lualatex/pdflatex on PATH -- .pgf files will NOT be written"
    echo "         (pdf/png/jpg will still be produced). Run this where latex is installed."
fi

# --- regenerate the four figures that live in reports/figures/pgf/ ---
"$PYTHON" src/energy_efficiency/error_sweep_training_triplet.py          # error_sweep_training_triplet.pgf
"$PYTHON" src/energy_efficiency/error_sweep_training_triplet_eepower.py  # error_sweep_training_triplet_eepower.pgf
"$PYTHON" src/energy_efficiency/ee_power_rate_tradeoff_sac.py            # ee_power_rate_tradeoff_sac_error0.pgf
"$PYTHON" src/energy_efficiency/plotting_scenario.py --plot-only         # error_sweep_sumrate.pgf

echo
echo "Done. Regenerated .pgf (+ pdf/png/jpg) under reports/figures/pgf/ :"
ls -la reports/figures/pgf/*.pgf
