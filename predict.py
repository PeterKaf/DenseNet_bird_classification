from config import *
import tensorflow as tf
import numpy as np
import os
import glob

# Load and preprocess the test dataset
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int',
    shuffle=False
)

# Initialize an empty list to hold the labels
labels_list = []

# Iterate over the dataset to collect labels
for images, labels in test_dataset:
    labels_list.extend(labels.numpy())

# Get the class names from the test dataset
class_names = test_dataset.class_names
test_dataset = test_dataset.map(preprocessing, num_parallel_calls=tf.data.AUTOTUNE)

# Reconstruct model architecture
model, start_epoch = compile_model()

# Save all filenames and randomly select subset of 20
all_files = sorted(set(f.numpy().decode('utf-8') for f in tf.data.Dataset.list_files(os.path.join(TEST_DIR, '*'))))
selected_labels = np.random.choice(all_files, size=20, replace=False)

# Initialize lists to hold the paths of the selected images and their corresponding labels
selected_image_paths = []
selected_actual_labels = []

for label in selected_labels:
    # List files in the label directory using glob
    files_in_label_dir = glob.glob(os.path.join(label, '*'))  # List files in the label directory

    # Ensure there are files in the directory before selecting a random file path
    if len(files_in_label_dir) > 0:
        # Select one random file path from the list
        selected_file_path = np.random.choice(files_in_label_dir, size=1)[0]  # Select one random file path
        selected_image_paths.append(selected_file_path)
        selected_actual_labels.append(label)
    else:
        print(f"No files found in directory: {label}")

pred = model.predict(test_dataset)
pred = np.argmax(pred, axis=1)


positive_count = 0
negative_count = 0

for i, item in enumerate(pred):
    print("Detected: ", item, "-", class_names[item], "True: ", labels_list[i], "-", class_names[labels_list[i]])
    if item == labels_list[i]:
        positive_count += 1
    else:
        negative_count += 1

print(f"Precision: {positive_count/(negative_count+positive_count)}")
