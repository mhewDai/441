#!/bin/bash

# Print a message to indicate the start of the script
echo "Running Homework 2 experiments..."

# Navigate to the source code directory
cd src

# Install the required Python packages
pip install -r requirements.txt

# Run the Python scripts that train the model and generate predictions
python3 task-123.py 
python3 task4.py    

echo "Experiments completed. Predictions saved."