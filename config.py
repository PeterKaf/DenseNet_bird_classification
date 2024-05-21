import os
from datetime import datetime

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
