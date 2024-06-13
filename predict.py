from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from config import *
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import glob


def preprocessing(image, label):
    image = preprocess_input(image)
    return image, label


# Load and preprocess the test dataset
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int',
    shuffle=False
)
test_dataset = test_dataset.map(preprocessing, num_parallel_calls=tf.data.AUTOTUNE)

# Reconstruct model architecture
model, start_epoch = compile_model()

# Step 1: List all unique labels
all_labels = sorted(set(f.numpy().decode('utf-8') for f in tf.data.Dataset.list_files(os.path.join(TEST_DIR, '*'))))
# Step 2: Randomly select 20 labels
selected_labels = np.random.choice(all_labels, size=20, replace=False)

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


# Assuming selected_image_paths and selected_actual_labels are defined elsewhere
num_images = len(selected_image_paths)

# Calculate the number of columns needed to fill 4 rows without exceeding the total number of images
cols_per_row = min(num_images // 4, 4)  # At most 4 columns per row
num_rows = -(-num_images // cols_per_row)  # Calculate the number of rows needed

# Create a figure with a grid of subplots arranged in 4 rows
fig, axs = plt.subplots(num_rows, cols_per_row, figsize=(15, 10))  # Adjust figsize as needed

# Flatten the array of axes for easier iteration
axs = axs.ravel()

# Load and display each image in its subplot
for i, ax in enumerate(axs):
    # Check if the current index exceeds the total number of images
    if i < num_images:
        # Load the image
        loaded_image = load_img(selected_image_paths[i], target_size=(224, 224))

        # Convert the loaded image to a NumPy array
        image_array = img_to_array(loaded_image)

        # Expand dimensions to match the model's input shape
        # Assuming your model expects input shape (batch_size, height, width, channels)
        image_array = np.expand_dims(image_array, axis=0)

        # Predict with the model
        predictions = model.predict(image_array)

        # Optionally, process the predictions to get the class name or confidence score
        pred_label = selected_actual_labels[i]
        # Determine title color based on match between predicted and actual labels
        title_color = 'green' if pred_label == selected_actual_labels[i] else 'red'

        # Display the image
        ax.imshow(loaded_image)
        ax.set_title(pred_label, color=title_color)
        ax.set_xlabel(selected_actual_labels[i], color='black')
        ax.axis('off')  # Hide axes
    else:
        # Remove the extra subplot if there are fewer images than the calculated number of subplots
        fig.delaxes(axs[i])

# Show the plot
plt.show()