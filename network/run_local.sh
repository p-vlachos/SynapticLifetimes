#!/bin/bash
# Runs the simulations in the current run folder (called by run.sh and run_test.sh).
# usage: run_local.sh <run name> <mode: run | test | net-test> [<parallel sims>] [<test dir>]

RUNNAME=$1
MODE=$2
NPARSIM=$3
TESTDIR=$4

echo ""
echo "Running $RUNNAME (mode: $MODE)"
echo ""

TESTFLAG=''
if [ "$MODE" != "run" ]; then
    TESTFLAG='-t'
fi

# print progress immediately (output goes to a pipe or file)
export PYTHONUNBUFFERED=1

PYTHONPATH="$PWD/src" python -m net.xstrct_run ${NPARSIM:+-c $NPARSIM} $TESTFLAG
status=$?
if [ $status -ne 0 ]; then
    echo "Error: the simulation failed (exit code $status), run folder left in $(pwd)"
    exit $status
fi

if [ "$MODE" == "net-test" ]; then

    # check the expected outputs of the test (the run folder stays in ../tests/testing/)
    export EIB_STANDALONE_TEST=1
    PYTHONPATH="$PWD/src" python "./src/${TESTDIR}tests.py" 2> test.log
    if [ $? -ne 0 ]; then
        echo "Error: test failed, see tests/testing/$RUNNAME/test.log"
    else
        echo "Test passed."
    fi

else

    if [ "$MODE" == "run" ]; then
        DESTINATION='completed'
    else
        DESTINATION='tests/man'
    fi

    cp src/run_analysis.sh .
    mkdir -p ../../$DESTINATION/
    mv "$(pwd)" ../../$DESTINATION/$RUNNAME
    echo "Done. Results in $DESTINATION/$RUNNAME"

fi
