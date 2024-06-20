from config import *
import tensorflow as tf

# Replace 'path_to_your_image.jpg' with the actual path to the image you want to predict
image_path = '//dataset/test/ABBOTTS BABBLER/155.jpg'

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int',
    shuffle=False
)

predict_image(image_path, test_dataset)
