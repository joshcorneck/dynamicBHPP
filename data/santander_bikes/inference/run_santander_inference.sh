#!/bin/bash

mkdir ../output/alpha
mkdir ../output/beta
mkdir ../output/group_memberships
mkdir ../output/tau

# Navigate to the root of the project
cd "$(dirname "$0")"/../../..
project_root=$(pwd)

# Set up the environment variables
export PYTHONPATH=${project_root}/

## Activate the virtual environment
# source online_networks_venv/bin/activate

echo "Starting inference, known graph and number of groups."
python3 data/santander_bikes/inference/santander_network_inference.py 
echo "Inference completed."

echo "Starting inference, unknown graph."
python3 data/santander_bikes/inference/santander_network_inference_inf_graph.py 
echo "Inference completed."

echo "Starting inference, unknown number of groups."
python3 data/santander_bikes/inference/santander_network_inference_inf_groups.py 
echo "Inference completed."