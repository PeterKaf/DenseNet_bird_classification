import os
from datetime import datetime
from tensorflow.keras.applications import DenseNet121
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
USE_WEIGHTS = False  # Set to true in case of retraining model from weights otherwise false


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
