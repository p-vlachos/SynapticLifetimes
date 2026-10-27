#!/bin/bash
# Creates (or updates) the Python environment for the simulations as a venv in
# network/myenv, e.g. on clusters without conda (alternative to environment.yml).
# Requires Python 3.12; to use another interpreter than python3:
#   PYTHON=python3.12 ./setup_env.sh
# Needs internet access, so run it on a login node before submitting jobs.

set -e
cd "$(dirname "$0")"

ENV_DIR="myenv"
PYTHON=${PYTHON:-python3}

# Create the virtual environment if it doesn't exist
if [ ! -d "$ENV_DIR" ]; then
    version=$($PYTHON -c 'import sys; print("%d.%d" % sys.version_info[:2])')
    if [ "$version" != "3.12" ]; then
        echo "error: $PYTHON is Python $version, but Python 3.12 is required" \
             "(e.g. load a python/3.12 module or set PYTHON=...)" >&2
        exit 1
    fi
    $PYTHON -m venv $ENV_DIR
fi

# Activate the environment, upgrade pip and install the requirements
source $ENV_DIR/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
deactivate

echo "Environment ready, activate it with: source network/$ENV_DIR/bin/activate"
