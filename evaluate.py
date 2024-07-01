# Import packages
import tensorflow as tf
from config import *

# Load and preprocess the test dataset
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int',
    shuffle=False
)
# Apply preprocessing to the test dataset
test_dataset = test_dataset.map(preprocessing, num_parallel_calls=tf.data.AUTOTUNE)

# Reconstruct model architecture
model, start_epoch = compile_model()

# Evaluate loaded model
test_loss, test_acc = model.evaluate(test_dataset)
print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')
