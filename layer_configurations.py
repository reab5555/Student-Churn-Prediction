# Generate 10 layer configurations for grid search
layer_configs = [
    [8, 16, 32],
    [16, 32, 64],
    [32, 64, 128],
    [8, 8, 8],
    [16, 16, 16],
    [32, 32, 32],
    [64, 64, 64],
    [128, 128, 128],
    [256, 128, 64],
    [8, 16, 64],
    [16, 32, 128]
]

learning_rates = [0.001, 0.01]  # Learning rates

# Save the layer configurations and learning rates to a file
import json

configurations = {
    "layer_configs": layer_configs,
    "learning_rates": learning_rates
}

with open("layer_configurations.json", "w") as f:
    json.dump(configurations, f)

print("Layer configurations and learning rates saved to 'layer_configurations.json'")
