#!/bin/bash
# Runs the post-processing (net/post_processing.py) of a finished run again, e.g. after
# changing the analysis code, with the code of the run (src/). From inside the run folder:
#
#   cd completed/<run>
#   ./src/post_process.sh [-n N]      (-n: at most N at a time)

if [ ! -d src/net ] || [ ! -d builds ]; then
    echo "error: run this from inside a run folder (with src/ and builds/)"
    exit 1
fi

PYTHONPATH=src python -m net.run --postprocess "$@"
