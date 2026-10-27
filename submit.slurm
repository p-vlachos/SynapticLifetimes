#!/bin/bash
#SBATCH --job-name=synlife
#SBATCH --output=synlife_%j.log
#SBATCH --nodes=1
#SBATCH --exclusive              # the whole node: all parameter sets run in parallel
#SBATCH --mem=0                  # all memory of the node (~4 GB per simulation with N_e=1600)
#SBATCH --time=12:00:00          # a full run (N_e=1600, T2=5000 s, T5=1500 s) takes ~3.5 h
##SBATCH --partition=<name>      # uncomment if your cluster requires a partition (see sinfo)

# Runs all parameter sets of network/net/explored_params.py on one node; submit from the
# repository root:
#   sbatch submit.slurm                  real run, results in completed/<timestamp>_<name>/
#   sbatch submit.slurm -t               test run, results in tests/man/
#   sbatch submit.slurm -D "description" (all options of network/run.sh can be added)
# Progress in synlife_<jobid>.log.
#
# Python environment: the venv of network/setup_env.sh if it exists, otherwise the conda
# environment synlife of the conda installation in CONDA_BASE (default ~/miniforge3).

set -e
cd "$SLURM_SUBMIT_DIR"
[ -d network ] && cd network

# module load ...                # add modules here if g++ or LaTeX need them

if [ -d myenv ]; then
    source myenv/bin/activate
else
    source "${CONDA_BASE:-$HOME/miniforge3}/bin/activate" synlife
fi

./run.sh "$@"
