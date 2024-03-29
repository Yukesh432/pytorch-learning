import os
import shutil
from glob import glob

def copy_config_images_by_criteria(experiments_directory, target_directory_name="selected_configs", **criteria):
    base_directory = os.path.dirname(os.path.abspath(__file__))
    source_directory = os.path.join(base_directory, experiments_directory)
    target_directory = os.path.join(base_directory, target_directory_name)
    
    os.makedirs(target_directory, exist_ok=True)
    
    # Build parts of the file name to search for based on criteria
    search_parts = [f"{key}_{value}" for key, value in criteria.items()]
    
    # Find all files in the source directory
    all_files = glob(os.path.join(source_directory, "*.png"))
    
    # Filter files that match all parts of the search criteria
    matched_files = [file for file in all_files if all(part in file for part in search_parts)]
    
    print(f"Found {len(matched_files)} files matching the criteria.")
    
    for file_path in matched_files:
        shutil.copy(file_path, target_directory)
        print(f"Copied {os.path.basename(file_path)} to {target_directory}")

# Example usage:
# To copy all images with ReLU activation function
# copy_config_images_by_criteria("experiments", activation_function="ReLU")

# To copy all images where initialization method is "random_normal"
# copy_config_images_by_criteria("experiments", initialization_method="ones")

# You can also combine criteria
# For example, to copy all images with ReLU activation function AND "random_normal" initialization
copy_config_images_by_criteria("experiments", activation_function="linear", initialization_method="ones")
