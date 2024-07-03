#!/bin/bash

# Make required directories
mkdir ../simulation_output/true_changes
mkdir ../simulation_output/tau
mkdir ../simulation_output/alpha
mkdir ../simulation_output/beta
mkdir ../simulation_output/group_changes
mkdir ../simulation_output/group_memberships

# Navigate to the root of the project
cd "$(dirname "$0")"/../../..
project_root=$(pwd)
echo "Changed to project root directory: ${project_root}"

# Set up the environment variables
export PYTHONPATH=${project_root}/

# # Activate the virtual environment
# source online_networks_venv/bin/activate

# Loop over the indices you want to run locally
for PBS_ARRAYID in {0..11}; do
    echo "Running simulation for parameter set ${PBS_ARRAYID}"
    python3 analyses/simulation_studies/simulation_parameters/simulation_study_3/simulation_study_3.py --index ${PBS_ARRAYID} 
done

echo "All simulations completed."
