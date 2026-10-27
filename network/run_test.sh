#!/bin/bash
# Runs a network test: the simulations of its explored_params.py, then its checks (tests.py).
#
#   ./run_test.sh tests/net_tests/test_standard_net
#
# Runs in ../tests/testing/<timestamp>_<test>/; the output is shown and saved in run.log,
# the results of the checks in test.log.

TEST=${1%/}
if [ ! -f "$TEST/tests.py" ]; then
    echo "usage: ./run_test.sh <test folder>, e.g. ./run_test.sh tests/net_tests/test_standard_net"
    exit 1
fi

RUN=$(date +%y%m%d_%H%M%S)_$(basename $TEST)
mkdir -p ../tests/testing/$RUN
rsync -a --exclude='__pycache__' --exclude='*~' ./ ../tests/testing/$RUN/src/
cp $TEST/explored_params.py ../tests/testing/$RUN/src/net/
cd ../tests/testing/$RUN
echo "Test folder: tests/testing/$RUN"

run() {
    PYTHONPATH=src PYTHONUNBUFFERED=1 python -m net.run || return 1
    if PYTHONPATH=src python src/$TEST/tests.py 2> test.log; then
        echo "Test passed."
    else
        echo "Test FAILED, see tests/testing/$RUN/test.log"
        return 1
    fi
}

run 2>&1 | tee run.log
exit ${PIPESTATUS[0]}
