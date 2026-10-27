#!/bin/bash

usage() {
    cat <<EOF
usage: ./run.sh [-t] [-b] [-n N] [-P name] [-D description]

Runs all parameter sets of net/explored_params.py, in a copy of the code in
../running/<timestamp>_<name>/, which is moved to ../completed/ when finished.
The git commit is recorded with the results; uncommitted changes of tracked
files are saved in uncommitted_changes.diff of the run folder. The complete
output is printed and saved in run.log of the run folder.

  -t          test run: no git record, results go to ../tests/man/
  -b          run in the background (output only in run.log of the run folder)
  -n N        run at most N simulations in parallel (default: all parameter
              sets, at most the number of available cores; each simulation
              needs ~4 GB of memory with N_e=1600)
  -P name     name of the run folder (default: name in net/explored_params.py)
  -D text     description, saved in the file 'description' of the run folder
  -h          show this help
EOF
}

MODE='run'
BACKGROUND=false
NPARSIM=''
NAME=''
DESCRIPTION=''

while getopts "tbn:P:D:h" opt; do
    case $opt in
    t) MODE='test' ;;
    b) BACKGROUND=true ;;
    n) NPARSIM=$OPTARG ;;
    P) NAME=$OPTARG ;;
    D) DESCRIPTION=$OPTARG ;;
    h) usage; exit 0 ;;
    *) usage; exit 1 ;;
    esac
done

CODEDIR=$(pwd)

if [ -z "$NAME" ]; then
    NAME=$(PYTHONPATH="$CODEDIR" python -c "from net.explored_params import name; print(name)") || exit 1
fi

# run folder: timestamp + name
RUNNAME=$(date +"%y%m%d_%H%M%S")_$NAME

mkdir -p ../running/$RUNNAME
rsync -a --exclude='*~' --exclude='__pycache__' --exclude='myenv/' \
      $CODEDIR/ ../running/$RUNNAME/src/

cd ../running/$RUNNAME

if [ -n "$DESCRIPTION" ]; then
    echo "$DESCRIPTION" > ./description
fi

echo "Run folder: running/$RUNNAME"
if [ -n "$SLURM_JOB_ID" ]; then
    echo "Slurm job $SLURM_JOB_ID on $(hostname)" > run.log
fi

# the complete output is saved in run.log of the run folder (moved with it)
if $BACKGROUND
then
    nohup ./src/run_local.sh $RUNNAME $MODE "$NPARSIM" >> run.log 2>&1 &
    echo "Running in the background, output in running/$RUNNAME/run.log"
else
    ./src/run_local.sh $RUNNAME $MODE "$NPARSIM" 2>&1 | tee -a run.log
    exit ${PIPESTATUS[0]}
fi
