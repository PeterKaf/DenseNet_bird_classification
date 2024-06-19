import os
import cv2
import numpy as np
from datetime import datetime
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Define paths
TRAIN_DIR = 'dataset/train'
VALIDATION_DIR = 'dataset/valid'
TEST_DIR = 'dataset/test'
CHECKPOINTS_DIR = "checkpoints"
LOG_DIR = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")  # Logs directory
MODEL_FILEPATH = "checkpoints/model-024-0.526.keras"  # Filepath of the model to load (Not used with weights approach)
MODEL_WEIGHTS = "checkpoints/weights/model.weights.h5"

# Define parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_CLASSES = len(os.listdir(TRAIN_DIR))
USE_WEIGHTS = True  # Set to true in case of retraining model from weights otherwise false


def compile_model(start_epoch=0):
    """
    Loads DenseNet121 model, adds custom layers, perform model compile depending on wheather or not you want to use
    weights
    :return: Compiled model, initial epoch number
    """
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

    if USE_WEIGHTS is True and os.path.exists(MODEL_WEIGHTS):
        print(f"Weights loading from : {MODEL_WEIGHTS}")
        model.load_weights(MODEL_WEIGHTS)

    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])
    return model, start_epoch


def get_class_names(test_dataset):
    class_names = test_dataset.class_names
    return class_names


def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, IMG_SIZE)  # Assuming IMG_SIZE is defined elsewhere
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = preprocess_input(img)
    return img


# Function to make predictions
def predict_image(image_path, test_dataset):
    # Reconstruct model architecture
    model, start_epoch = compile_model()

    # Load and preprocess the image
    image = load_and_preprocess_image(image_path)

    # Make prediction
    pred = model.predict(image)
    predicted_class = int(np.argmax(pred, axis=1))
    accuracy = np.max(pred)
    class_names = get_class_names(test_dataset)

    # Print the prediction result
    print(f"Predicted Class: {predicted_class}, Class Name: {class_names[predicted_class]}, Accuracy: {accuracy}")
