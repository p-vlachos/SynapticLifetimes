#!/bin/bash

usage() {
    cat <<EOF
usage: ./src/post_process.sh [-n N]     (from inside a run folder, e.g. completed/<run>/)

Runs the post-processing (net/post_processing.py: survival and lifetime statistics,
turnover counts, branching ratio and survival figures) again for all simulations of a
finished run, with the code of the run (src/).

  -n N    run at most N post-processings in parallel (default: all simulations,
          at most the number of available cores)
  -h      show this help
EOF
}

NPARSIM=''

while getopts "n:h" opt; do
    case $opt in
    n) NPARSIM=$OPTARG ;;
    h) usage; exit 0 ;;
    *) usage; exit 1 ;;
    esac
done

if [ ! -d src/net ] || [ ! -d builds ]; then
    echo "error: run this from inside a run folder (with src/ and builds/)" >&2
    exit 1
fi

PYTHONPATH="$PWD/src" python -m net.xstrct_run -x -t ${NPARSIM:+-c $NPARSIM}
