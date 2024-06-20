"""
After using dataset_merger, we get all data combined in converted_dataset. This script access said dataset and splits it
to specified ratio
"""
import splitfolders

# Specify the input folder where your original directories are located
input_folder = "converted_dataset"

# Specify the output folder where you want the split datasets to be saved
output = "dataset"

# Use the split_folders.ratio function to split the dataset
splitfolders.ratio(input_folder, output=output, seed=42, ratio=(0.8, 0.1, 0.1))
