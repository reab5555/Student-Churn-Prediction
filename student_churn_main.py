import os
import subprocess

# Define the scripts to run in order
scripts = [
    "data_preprocessing.py",
    "feature_selection.py",
    "simple_model_evaluation.py",
    "feature_importance.py",
    "layer_configurations.py",
    "neural_network.py"
]

# Execute each script in order
for script in scripts:
    print(f"Running {script}...")
    subprocess.run(["python", script], check=True)
    print(f"Finished running {script}.\n")

print("All scripts executed successfully.")


