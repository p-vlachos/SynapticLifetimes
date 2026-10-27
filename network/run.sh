#!/bin/bash

usage() {
    cat <<EOF
usage: ./run.sh [-n N] [-b]

Runs all parameter sets of net/explored_params.py, several in parallel.

The code is copied to ../running/<timestamp>_<name>/ (name: set in explored_params.py)
and the simulations run there. When they have finished, the folder is moved to
../completed/. The output is shown and saved in run.log of that folder, together with
the git commit (git_commit.txt) and any uncommitted changes (uncommitted_changes.diff).

  -n N   run at most N simulations at a time (default: all, at most one per core;
         each simulation needs ~4 GB of memory with N_e=1600)
  -b     run in the background (output only in run.log)
EOF
}

NPARSIM=""
BACKGROUND=false
while getopts "n:bh" opt; do
    case $opt in
    n) NPARSIM=$OPTARG ;;
    b) BACKGROUND=true ;;
    *) usage; exit 1 ;;
    esac
done

# run folder: timestamp + name from explored_params.py
NAME=$(PYTHONPATH=. python -c "from net.explored_params import name; print(name)") || exit 1
RUN=$(date +%y%m%d_%H%M%S)_$NAME
mkdir -p ../running/$RUN

# copy the code and record its git state
rsync -a --exclude='__pycache__' --exclude='*~' ./ ../running/$RUN/src/
git log -1 --format="%H %s" > ../running/$RUN/git_commit.txt 2>/dev/null
git diff HEAD > ../running/$RUN/uncommitted_changes.diff 2>/dev/null
for f in git_commit.txt uncommitted_changes.diff; do
    [ -s ../running/$RUN/$f ] || rm ../running/$RUN/$f    # remove if empty
done

cd ../running/$RUN
echo "Run folder: running/$RUN"

run() {
    [ -n "$SLURM_JOB_ID" ] && echo "Slurm job $SLURM_JOB_ID on $(hostname)"
    if ! PYTHONPATH=src PYTHONUNBUFFERED=1 python -m net.run ${NPARSIM:+-n $NPARSIM}; then
        echo "Failed: the run folder stays in running/$RUN"
        return 1
    fi
    mkdir -p ../../completed
    mv ../$RUN ../../completed/
    echo "Done: completed/$RUN"
}

if $BACKGROUND; then
    run > run.log 2>&1 &
    echo "Running in the background, output in running/$RUN/run.log"
else
    run 2>&1 | tee run.log
    exit ${PIPESTATUS[0]}
fi
