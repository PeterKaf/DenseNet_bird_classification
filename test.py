# Import packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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

# Define logs directory
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Define checkpoints directory
checkpoints_dir = "checkpoints"

# Data generators - preprocess_input is now used here
train_datagen = ImageDataGenerator(rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest',
                                   preprocessing_function=preprocess_input)

validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=IMG_SIZE,
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                              target_size=IMG_SIZE,
                                                              batch_size=BATCH_SIZE,
                                                              class_mode='categorical')

test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=IMG_SIZE,
                                                  batch_size=BATCH_SIZE,
                                                  class_mode='categorical',
                                                  shuffle=False)

# Load the DenseNet model
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model
base_model.trainable = False

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

# Final model
model = Model(inputs=base_model.input, outputs=predictions)

# See the layout
model.summary()

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
history = model.fit(train_generator,
                    steps_per_epoch=(train_generator.samples // BATCH_SIZE)-1,
                    epochs=20,
                    validation_data=validation_generator,
                    validation_steps=(validation_generator.samples // BATCH_SIZE)-1,
                    callbacks=[tensorboard_callback, checkpoint_callback, early_stopping_callback])

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator, steps=(test_generator.samples // BATCH_SIZE)-1)
print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')

# Save the model in TensorFlow SavedModel format
model.save('trained_models/model_4.keras')
