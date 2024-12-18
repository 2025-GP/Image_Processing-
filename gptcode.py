import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K

# Constants for dataset
DATASET_DIR = 'IPProjectDataset24'
IMAGE_SIZE = (576, 576)
NUM_CLASSES = 8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Ensure this line is executed before importing TensorFlow
os.environ['OMP_NUM_THREADS'] = '1'  # Limit the number of threads
os.environ["TF_CPU_THREADPOOL_SIZE"] = "1"  # Limit CPU threadpool size
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'  # Disable XLA

tf.config.set_visible_devices([], 'GPU')  # This forces TensorFlow to use only the CPU
K.clear_session()  # Clear session to reset the model graph

# Label color mappings
LABEL_COLORS = {
    (0, 0, 0): 0,
    (128, 0, 0): 1,
    (128, 64, 128): 2,
    (0, 128, 0): 3,
    (128, 128, 0): 4,
    (64, 0, 128): 5,
    (192, 0, 192): 6,
    (64, 64, 0): 7,
}

# Function to map RGB to class index
def rgb_to_class(label):
    label = np.asarray(label, dtype=np.uint8)
    class_map = np.zeros(label.shape[:2], dtype=np.uint8)
    for rgb, cls in LABEL_COLORS.items():
        mask = (label[:, :, 0] == rgb[0]) & (label[:, :, 1] == rgb[1]) & (label[:, :, 2] == rgb[2])
        class_map[mask] = cls
    return class_map

# Function to load dataset
def load_data(data_dir):
    images = []
    labels = []

    image_dir = os.path.join(data_dir, 'Images')
    label_dir = os.path.join(data_dir, 'Labels')

    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(label_dir, image_name)  # Assuming same names for images and labels

        image = tf.keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
        image = tf.keras.utils.img_to_array(image) / 255.0

        label = tf.keras.utils.load_img(label_path, target_size=IMAGE_SIZE)
        label = tf.keras.utils.img_to_array(label)
        label = rgb_to_class(label)
        label = to_categorical(label, num_classes=NUM_CLASSES)

        images.append(image)
        labels.append(label)

    return np.array(images), np.array(labels)

# Load training and validation data
train_images, train_labels = load_data(os.path.join(DATASET_DIR, 'train_data'))
val_images, val_labels = load_data(os.path.join(DATASET_DIR, 'val_data'))

# U-Net model definition
def encoder_block(inputs, num_filters):
    x = tf.keras.layers.Conv2D(num_filters, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(num_filters, 3, padding='same', activation='relu')(x)
    p = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    return x, p

def decoder_block(inputs, skip_features, num_filters):
    x = tf.keras.layers.Conv2DTranspose(num_filters, 3, strides=2, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Concatenate()([x, skip_features])  # Concatenate should handle the dimensions
    x = tf.keras.layers.Conv2D(num_filters, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(num_filters, 3, padding='same', activation='relu')(x)
    return x

def unet_model(input_shape, num_classes):
    inputs = tf.keras.layers.Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = tf.keras.layers.Conv2D(1024, 3, padding='same', activation='relu')(p4)
    b1 = tf.keras.layers.Conv2D(1024, 3, padding='same', activation='relu')(b1)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = tf.keras.layers.Conv2D(num_classes, 1, activation='softmax')(d4)

    model = tf.keras.models.Model(inputs, outputs)
    return model

# Model compilation
model = unet_model(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), num_classes=NUM_CLASSES)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model training
model.fit(
    train_images, train_labels,
    validation_data=(val_images, val_labels),
    batch_size=4,
    epochs=25
)

# Save the model
model.save('unet_model.h5')
