# Import packages
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from config import *

"""
# Define parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
model_weights = 'trained_models/model.weights.h5'
test_dir = 'dataset/test'
num_classes = len(os.listdir(test_dir))
"""
# Load and preprocess the test dataset


def preprocessing(image, label):
    image = preprocess_input(image)
    return image, label


test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int',
    shuffle=False
)
# Apply preprocessing to the test dataset
test_dataset = test_dataset.map(preprocessing, num_parallel_calls=tf.data.AUTOTUNE)

# Load weights
# Load the DenseNet model
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

# Final model
model = Model(inputs=base_model.input, outputs=predictions)


model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

print(f"Weights loaded from : {MODEL_WEIGHTS}")
model.load_weights(MODEL_WEIGHTS)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_dataset)
print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')
