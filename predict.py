from config import *
import tensorflow as tf
import numpy as np

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

# Make predictions
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
