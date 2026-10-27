#!/bin/bash
#SBATCH --job-name=net_test
#SBATCH --output=net_test_output.log
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=2G


# Step 0: Go to network folder
cd network

# Step 1: Set up the Python environment
bash setup_env.sh
source myenv/bin/activate

# Step 2: Run the simulation test
./run_test.sh -l tests/net_tests/test_standard_net/