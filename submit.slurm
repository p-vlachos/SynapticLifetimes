#!/bin/bash
#SBATCH --job-name=synlife
#SBATCH --output=synlife_%j.log
#SBATCH --nodes=1
#SBATCH --exclusive              # the whole node: all parameter sets run in parallel
#SBATCH --mem=0                  # all memory of the node (~4 GB per simulation with N_e=1600)
#SBATCH --time=12:00:00          # a full run (N_e=1600, T2=5000 s, T5=1500 s) takes ~3.5 h
#SBATCH --partition=general1,general2      # partitions of the Goethe cluster (see sinfo)

# Runs all parameter sets of network/net/explored_params.py on one node. Can be submitted from
# any folder, e.g. from a folder submitted/ in the repository:
#   cd submitted
#   sbatch ../submit.slurm                  real run, results in completed/<timestamp>_<name>/
#   sbatch ../submit.slurm -t               test run, results in tests/man/
#   sbatch ../submit.slurm -D "description" (all options of network/run.sh can be added)
# The job log synlife_<jobid>.log in the submission folder shows the progress and errors;
# all data and the complete log (run.log) are saved in the run folder in the repository.
#
# Python environment: the venv of network/setup_env.sh if it exists, otherwise the conda
# environment synlife of the conda installation in CONDA_BASE (default ~/miniforge3).

set -e

# The repository: SYNLIFE_DIR if set, otherwise the first folder containing network/run.sh
# among the folder of this script (sbatch runs a copy, the original path is in the job
# info) and the submission folder, and their parent folders
find_repository() {
    local dir
    for dir in "$@"; do
        while [ -n "$dir" ] && [ "$dir" != "/" ]; do
            if [ -x "$dir/network/run.sh" ]; then
                echo "$dir"
                return
            fi
            dir=$(dirname "$dir")
        done
    done
}
script=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | sed -n 's/^ *Command=\([^ ]*\).*/\1/p')
REPO=${SYNLIFE_DIR:-$(find_repository "$(dirname "$script")" "$SLURM_SUBMIT_DIR")}
if [ -z "$REPO" ]; then
    echo "error: repository not found (submitted from $SLURM_SUBMIT_DIR, script $script);" \
         "submit from the repository or set SYNLIFE_DIR, e.g." \
         "sbatch --export=ALL,SYNLIFE_DIR=\$HOME/projects/SynapticLifetimes submit.slurm" >&2
    exit 1
fi
cd "$REPO/network"
echo "Repository: $REPO"

# module load ...                # add modules here if g++ or LaTeX need them

if [ -d myenv ]; then
    source myenv/bin/activate
else
    source "${CONDA_BASE:-$HOME/miniforge3}/bin/activate" synlife
fi

./run.sh "$@"
