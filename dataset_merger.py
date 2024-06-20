"""
This script collects data from train / test / valid directories, and puts them into single directory while keeping
subdirectory name with labels. It should be used with dataset_redistribution script to get the datasplit ratio you want
"""
import os
import shutil

source_dirs = ['dataset/train', 'dataset/test', 'dataset/valid']
target_dir = 'converted_dataset'

subdirs = set()

for source_dir in source_dirs:
    for root, dirs, files in os.walk(source_dir):
        for directory in dirs:
            subdirs.add(directory)

for subdir in subdirs:
    # Create the target subdirectory
    target_subdir_path = os.path.join(target_dir, subdir)
    os.makedirs(target_subdir_path, exist_ok=True)

    # Initialize a counter for renaming images
    image_counter = 1

    for source_dir in source_dirs:
        source_subdir_path = os.path.join(source_dir, subdir)
        if os.path.exists(source_subdir_path):
            for file in os.listdir(source_subdir_path):
                if file.endswith('.jpg'):
                    # Construct the full file path
                    source_file_path = os.path.join(source_subdir_path, file)
                    # Construct the new file name
                    new_file_name = f"{image_counter}.jpg"
                    target_file_path = os.path.join(target_subdir_path, new_file_name)
                    # Copy and rename the file
                    shutil.copy(source_file_path, target_file_path)
                    # Increment the counter
                    image_counter += 1
