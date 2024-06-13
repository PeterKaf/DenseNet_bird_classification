import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.densenet import preprocess_input
from config import *
import matplotlib.pyplot as plt
import os
import glob


# Define a function to load and preprocess each image
def load_and_preprocess_image(path):
    # Load the image file
    image = tf.io.read_file(path)

    # Decode the image
    image = tf.image.decode_jpeg(image)

    # Resize the image to the desired size
    image = tf.image.resize(image, IMG_SIZE)

    # Preprocess the image (this step depends on your model; adjust accordingly)
    image = preprocess_input(image)

    # Return the preprocessed image
    return image


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

# Predict labels for the selected images
# Create a TensorFlow dataset from the list of image paths
sampled_images = tf.data.Dataset.from_tensor_slices(selected_image_paths)
# Map the load_and_preprocess_image function over the dataset
sampled_images = sampled_images.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
print('wait here')
sampled_images = sampled_images.batch(BATCH_SIZE)
predictions = model.predict(sampled_images)

# Display images with predicted vs actual labels
fig, axs = plt.subplots(4, 5, figsize=(15, 12))
axs = axs.ravel()  # Flatten the array of axes for easier iteration

for i, ax in enumerate(axs):
    if i >= len(selected_actual_labels):  # Break loop if we have fewer images than expected
        break

    # Predict label directly using np.argmax and mapping to actual labels
    predicted_index = np.argmax(predictions[i])
    pred_label = selected_actual_labels[i]  # Use the actual label associated with the selected image

    # Determine title color based on match between predicted and actual labels
    title_color = 'green' if pred_label == selected_actual_labels[i] else 'red'

    # Load and display image
    img = tf.io.read_file(selected_image_paths[i])
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, [IMG_SIZE[0], IMG_SIZE[1]])
    ax.imshow(img.numpy())
    ax.set_title(pred_label, color=title_color)
    ax.set_xlabel(selected_actual_labels[i], color='black')
    ax.axis('off')

plt.tight_layout()
plt.show()
