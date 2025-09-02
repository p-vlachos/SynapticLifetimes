#!/bin/bash

# Set environment directory name
ENV_DIR="myenv"

# Create the virtual environment if it doesn't exist
if [ ! -d "$ENV_DIR" ]; then
    python3 -m venv $ENV_DIR
fi

# Activate the environment
source $ENV_DIR/bin/activate

# Upgrade pip and install requirements
pip install --upgrade pip
pip install -r requirements.txt
pip install git+https://github.com/excinhbal/brian2@2.3.0.2_excinhbal_fixes#egg=Brian

# (Optional) Deactivate environment
deactivate
