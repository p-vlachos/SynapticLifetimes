#!/bin/bash
#SBATCH --job-name=synlife
#SBATCH --output=synlife_%j.log      # job log, in the folder you submit from
#SBATCH --partition=general1,general2
#SBATCH --nodes=1
#SBATCH --exclusive                  # the whole node: all parameter sets run in parallel
#SBATCH --mem=0                      # all memory of the node (~4 GB per simulation, N_e=1600)
#SBATCH --time=5-12:00:00            # a full run (N_e=1600, T2=5000 s, T5=1500 s) takes ~3.5 h

# Runs network/run.sh, i.e. all parameter sets of network/net/explored_params.py, on one node.
# Submit from the folder where the job log should go, e.g.:
#
#   cd ~/submitted
#   sbatch ~/projects/SynapticLifetimes/submit.slurm      (options of run.sh can be added, e.g. -n 4)
#
# The job log shows the progress and errors; the results and the complete log (run.log)
# are saved in completed/<timestamp>_<name>/ of the repository.

# the repository (adapt if it is somewhere else)
REPO=$HOME/projects/SynapticLifetimes
if [ ! -f "$REPO/network/run.sh" ]; then
    echo "error: repository not found at $REPO (see REPO in submit.slurm)"
    exit 1
fi

# the conda environment (see README); adapt the path if conda is installed elsewhere
source "$HOME/miniforge3/bin/activate" synlife

cd "$REPO/network"
./run.sh "$@"
