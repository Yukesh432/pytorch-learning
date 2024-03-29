import os
import shutil
from glob import glob

def copy_config_images(configs, experiments_directory, target_directory_name="selected_configs"):
    """
    Copies images matching given configurations from the 'experiments' directory to a 'selected_configs'
    directory, both of which are in the same directory as this script.

    :param configs: List of dictionaries, where each dictionary contains configuration parameters for the images.
    :param experiments_directory: The directory within the same directory as this script where the images are located.
    :param target_directory_name: The name of the new directory to which the files will be copied. Defaults to "selected_configs".
    """
    # Get the base directory of this script
    base_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the source (experiments) and target (selected_configs) directories' absolute paths
    source_directory = os.path.join(base_directory, experiments_directory)
    target_directory = os.path.join(base_directory, target_directory_name)
    
    # Create the target directory if it doesn't exist
    os.makedirs(target_directory, exist_ok=True)

    for config in configs:
        # Build the search pattern from the configuration
        pattern = "model_performance_" + "_".join([f"{key}_{value}" for key, value in config.items()]) + "_learning_curves.png"
        pattern = os.path.join(source_directory, pattern)
        
        # Find files that match the pattern
        for file_path in glob(pattern):
            # Copy each matching file to the target directory
            shutil.copy(file_path, target_directory)
            print(f"Copied {os.path.basename(file_path)} to {target_directory}")

# Define your configurations here
configs = [
    {
        "num_of_hidden_layers": 1,
        "num_of_hidden_units": 512,
        "learning_rate": 0.001,
        "optims": "SGD",
        "activation_function": "Selu",
        "initialization_method": "he_uniform",
        "dropout_percentage": "None",
        "batch_size": 64,
        "epochs": 50000,
        "patience": 10
    },
    {
        "num_of_hidden_layers": 1,
        "num_of_hidden_units": 32,
        "learning_rate": 0.001,
        "optims": "SGD",
        "activation_function": "ReLU",
        "initialization_method": "he_uniform",
        "dropout_percentage": "None",
        "batch_size": 64,
        "epochs": 50000,
        "patience": 10
    }
    # Add more configurations as needed
]

# This should be just the folder name, not a path, since it's located in the same directory as this script.
experiments_directory = "experiments"
copy_config_images(configs, experiments_directory)
