import os
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import datetime

# Define paths
train_dir = 'dataset/train'
validation_dir = 'dataset/valid'
test_dir = 'dataset/test'

# Define parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
num_classes = len(os.listdir(train_dir))

# Define logs directory
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Define checkpoints directory
checkpoints_dir = "checkpoints"


# Define preprocessing function
def preprocessing(image, label):
    image = preprocess_input(image)
    return image, label


# Load and preprocess the training dataset
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)


# Load and preprocess the datasets
validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    validation_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

# Load and preprocess the test dataset
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=False
)

# Apply preprocessing to the training dataset
train_dataset = train_dataset.map(preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
# Apply preprocessing to the validation dataset
validation_dataset = validation_dataset.map(preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
# Apply preprocessing to the test dataset
test_dataset = test_dataset.map(preprocessing, num_parallel_calls=tf.data.AUTOTUNE)

# Load the DenseNet model
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model
base_model.trainable = False

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)  # Use num_classes here

# Final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

checkpoint_callback = ModelCheckpoint(filepath=f"{checkpoints_dir}/model-{{epoch:03d}}-{{val_loss:.3f}}.keras",
                                      monitor='val_loss',
                                      verbose=1,
                                      save_best_only=True,
                                      mode='auto')

early_stopping_callback = EarlyStopping(monitor="val_loss",
                                        patience=5,
                                        restore_best_weights=True)

# Train the model with TensorBoard callback
history = model.fit(train_dataset,
                    epochs=20,
                    validation_data=validation_dataset,
                    callbacks=[tensorboard_callback, checkpoint_callback, early_stopping_callback])

# Evaluate the model
test_loss, test_acc = model.evaluate(test_dataset)
print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')

# Save the model in TensorFlow SavedModel format
model.save('trained_models/model_4.keras')
