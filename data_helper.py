"""

"""
import numpy as np
from huaytools.dataset import load_mnist as _load_mnist


def load_mnist():
    """"""
    (x_train, y_train), (x_test, y_test) = _load_mnist()

    image_size = x_train.shape[1]
    original_dim = image_size * image_size
    x_train = np.reshape(x_train, [-1, original_dim])
    x_test = np.reshape(x_test, [-1, original_dim])
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    return x_train, x_test, original_dim

