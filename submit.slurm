#!/bin/bash
#SBATCH --job-name=synlife
#SBATCH --output=synlife_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
##SBATCH --partition=<name>    # uncomment if your cluster requires a partition (see sinfo)

# Runs the network simulation inside this Slurm job; submit from the repository root:
#   sbatch submit.slurm
#
# Before the first job, create the Python environment on a login node (needs internet):
#   PYTHON=python3.12 network/setup_env.sh
#
# By default this job runs the standard network test. For a simulation with the parameters
# in network/net/explored_params.py, commit them and replace the last line with e.g.
#   ./run.sh -l -d -E -n 2 -P _myrun
# (-n: simulations in parallel, one per parameter combination). Then request
#   --cpus-per-task >= n, --mem ~4G per parallel simulation (N_e=1600, ~6G for N_e=3200),
#   --time ~3 h per simulation with the default parameters (N_e=1600, T2=5000 s, T5=1500 s).
# The results end up in completed/<timestamp>_myrun/.

set -e
cd "$SLURM_SUBMIT_DIR/network"

# Python environment from setup_env.sh (with conda, use `conda activate synlife` instead)
source myenv/bin/activate

# -l: compute inside this job (no further srun), -d: stay in the foreground until the end
./run_test.sh -l -d tests/net_tests/test_standard_net/
