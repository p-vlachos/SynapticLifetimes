#!/bin/bash
#SBATCH --job-name=synlife
#SBATCH --output=synlife_%j.log      # job log, in the folder you submit from
#SBATCH --partition=general1,general2
#SBATCH --nodes=1
#SBATCH --exclusive                  # the whole node: all parameter sets run in parallel
#SBATCH --mem=0                      # all memory of the node (~4 GB per simulation, N_e=1600)
#SBATCH --time=12:00:00              # a full run (N_e=1600, T2=5000 s, T5=1500 s) takes ~3.5 h

# Runs network/run.sh, i.e. all parameter sets of network/net/explored_params.py, on one node.
# Submit from the repository or a folder inside it, e.g. submitted/:
#
#   cd SynapticLifetimes/submitted
#   sbatch ../submit.slurm             (options of run.sh can be added, e.g. -n 4)
#
# The job log shows the progress and errors; the results and the complete log (run.log)
# are saved in completed/<timestamp>_<name>/ of the repository.

# the repository: the submission folder or the first folder above it with network/run.sh
REPO=$SLURM_SUBMIT_DIR
while [ ! -f "$REPO/network/run.sh" ]; do
    if [ "$REPO" = "/" ]; then
        echo "error: submit from the repository or a folder inside it"
        exit 1
    fi
    REPO=$(dirname "$REPO")
done

# the conda environment (see README); adapt the path if conda is installed elsewhere
source "$HOME/miniforge3/bin/activate" synlife

cd "$REPO/network"
./run.sh "$@"
