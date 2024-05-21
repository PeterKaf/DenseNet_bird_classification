import os
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from config import *


# Define preprocessing function
def preprocessing(image, label):
    image = preprocess_input(image)
    return image, label


# Load and preprocess the training dataset
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
)

# Load and preprocess the datasets
validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    VALIDATION_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
)

# Load and preprocess the test dataset
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int',
    shuffle=False
)

# Apply preprocessing to the training dataset
train_dataset = train_dataset.map(preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
# Apply preprocessing to the validation dataset
validation_dataset = validation_dataset.map(preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
# Apply preprocessing to the test dataset
test_dataset = test_dataset.map(preprocessing, num_parallel_calls=tf.data.AUTOTUNE)

# Define callbacks
tensorboard_callback = TensorBoard(log_dir=LOG_DIR, histogram_freq=1)

checkpoint_callback = ModelCheckpoint(filepath=f"{CHECKPOINTS_DIR}/model-{{epoch:03d}}-{{val_loss:.3f}}.keras",
                                      monitor='val_loss',
                                      verbose=1,
                                      save_best_only=True,
                                      mode='auto')

early_stopping_callback = EarlyStopping(monitor="val_loss",
                                        patience=5,
                                        restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor="val_loss",
                              factor=0.2,
                              patience=3,
                              min_lr=1e-6
                              )


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

if os.path.exists(MODEL_WEIGHTS):
    print(f"Weights loaded from : {MODEL_WEIGHTS}")
    model.load_weights(MODEL_WEIGHTS)

model.compile(optimizer=Adam(learning_rate=0.000008), loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])
start_epoch = 48

# Train the model with TensorBoard callback
history = model.fit(train_dataset,
                    epochs=100,
                    validation_data=validation_dataset,
                    callbacks=[tensorboard_callback, checkpoint_callback, early_stopping_callback, reduce_lr],
                    initial_epoch=start_epoch)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_dataset)
print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')

# Save the model in TensorFlow SavedModel format
model.save('trained_models/data_fixed.keras')
