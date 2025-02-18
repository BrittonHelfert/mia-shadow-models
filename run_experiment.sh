#!/bin/bash

# Activate virtual environment
source /data/$USER/mia_project/mia_env/bin/activate

# Set PYTHONPATH to include your project directory
export PYTHONPATH=/data/$USER/mia_project:$PYTHONPATH

# Run the experiment
# Remove --test_mode for the full experiment
python -m experiments.cifar10 \
    --target_epochs=100 \
    --attack_epochs=100 \
    --num_shadows=100

# Deactivate virtual environment
deactivate 