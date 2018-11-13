""" 
    Application of keras implementation of MobileNet trained on ImageNet.
    The MobileNetV2 model takes 14Mb of space.
    https://keras.io/applications/#mobilenet

    Each model must provide:
      * preprocessing given a file path

    TODO: also provide import
      * preprocessing given a PIL Image
"""

import numpy as np

from . import keras
from keras.preprocessing import image
from keras.applications.mobilenetv2 import MobileNetV2 as MobileNet
from keras.applications.mobilenetv2 import preprocess_input, decode_predictions


def fully_connected_model():
    """ Return pre-trained mobilenet Keras Model with fully connected layers. """
    model = MobileNet(  input_shape=None
                      , alpha=1.4
                      , depth_multiplier=1
                      , include_top=True
                      , weights='imagenet'
                      , input_tensor=None
                      , pooling='max'
                      , classes=1000)

    def _load_from_file(input_image_path):
        img = image.load_img(input_image_path, target_size=(model.input_shape[2], model.input_shape[1]))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        return img

    model.load_image_file = _load_from_file
    model.preprocess_input = preprocess_input
    model.decode_predictions = decode_predictions
    return model


def headless_model(input_shape):
    """ Return pre-trained mobilenet Keras Model with no top layers. """
    model = MobileNet( input_shape=input_shape
                     , alpha=1.0
                     , depth_multiplier=1
                     #, dropout=1e-3
                     , include_top=False
                     , weights='imagenet'
                     , input_tensor=None
                     , pooling='max' )

    model.preprocess_input = preprocess_input
    return model

