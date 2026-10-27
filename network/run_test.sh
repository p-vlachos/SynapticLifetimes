#!/bin/bash

usage() {
    cat <<EOF
usage: ./run_test.sh [-b] <test dir>     e.g. ./run_test.sh tests/net_tests/test_standard_net/

Runs the simulations of the test (its explored_params.py) in a copy of the code in
../tests/testing/<timestamp>_<test>/ and checks the results with its tests.py
(output in test.log of that folder).

  -b    run in the background (output only in run.log of the test folder)
  -h    show this help
EOF
}

BACKGROUND=false

while getopts "bh" opt; do
    case $opt in
    b) BACKGROUND=true ;;
    h) usage; exit 0 ;;
    *) usage; exit 1 ;;
    esac
done

TESTDIR_FULL=${@:$OPTIND:1}
if [ -z "$TESTDIR_FULL" ]; then
    usage; exit 1
fi
# with a trailing slash, as used for the paths below
TESTDIR_FULL=${TESTDIR_FULL%/}/
TESTDIR=$(basename $TESTDIR_FULL)

CODEDIR=$(pwd)

# test folder: timestamp + test name
RUNNAME=$(date +"%y%m%d_%H%M%S")_$TESTDIR

mkdir -p ../tests/testing/$RUNNAME
rsync -a --exclude='*~' --exclude='__pycache__' --exclude='myenv/' \
      $CODEDIR/ ../tests/testing/$RUNNAME/src/

cd ../tests/testing/$RUNNAME

# parameters of the test
cp ./src/${TESTDIR_FULL}explored_params.py ./src/net/

echo "Test folder: tests/testing/$RUNNAME"
# the complete output is saved in run.log of the test folder
if $BACKGROUND
then
    nohup ./src/run_local.sh $RUNNAME net-test "" $TESTDIR_FULL > run.log 2>&1 &
    echo "Running in the background, output in tests/testing/$RUNNAME/run.log"
else
    ./src/run_local.sh $RUNNAME net-test "" $TESTDIR_FULL 2>&1 | tee run.log
    exit ${PIPESTATUS[0]}
fi
