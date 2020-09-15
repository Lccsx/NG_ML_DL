import tensorflow as tf
import numpy as np
from tensorflow import keras
print(tf.__version__)

mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()