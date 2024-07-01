from config import *
import tensorflow as tf

# Provide path to the image you want to be classified (I used test set image, but it can be any image as long as it's
# not from train or valid set)
image_path = '//dataset/test/ABBOTTS BABBLER/155.jpg'

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int',
    shuffle=False
)

predict_image(image_path, test_dataset)
