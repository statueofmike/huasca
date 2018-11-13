

import numpy as np
import os

from ..root import keras
from keras.models import model_from_json as _model_from_json
import tensorflow as tf

_path = os.path.abspath(os.path.dirname(__file__))

def _load_demog_model():
    #print("Loading demographics models...")
    with open(_path+'/../bin/gender.classify.json','r') as f:
        json = f.read()
    gender_model = _model_from_json(json)
    gender_model.load_weights(_path+'/../bin/gender.classify.h5')
    #print("  - gender model loaded")

    with open(_path+'/../bin/age.classify.json','r') as f:
        json = f.read()
    age_model = _model_from_json(json)
    age_model.load_weights(_path+'/../bin/age.classify.h5')
    #print("  - age model loaded")

    #print("Demographics models loaded.")
    return gender_model, age_model

class Demographics:
    def __init__(self):
        self.graph = tf.get_default_graph()
        self.gender_model, self.age_model = _load_demog_model()


    def _predict(self,model,image,batch_size):
        with self.graph.as_default():
            return model.predict(image, batch_size=batch_size)

    def _cleanup(self):
        self.batch_size = 1
        self.gender_target_size = (100,100)
        self.age_target_size = (128, 128)
        self.gender_filter = {0:'male',1:'female'}
        self.age_filter = {0: '(0, 2)', 1:'(4, 6)' , 2:'(8, 12)', 3:'(15, 20)',4:'(25, 32)',5:'(38, 43)', 6:'(38, 43)', 7:'(38, 43)'}

        self.x_test = None
        self.preprocessed = None
        self.y_test = None
        self.scores = None
        self.predictions = None

    def _input_preprocessing(self,image,image_w, image_h):
        """ Preprocessing to match the training conditions for this model.
        Apply resize, reshape, other scaling/whitening effects.
        image can be any image size greater than 100x100 and it will be resized
        """
        image = image.resize((image_w,image_h))
        image = np.asarray(image)
        image = image * (1./255.)
        return image.reshape(1,image_w,image_h,3)


    def classify(self,x_test):
        self._cleanup()
        
        # Gender Prediction
        w, h = self.gender_target_size[0], self.gender_target_size[1]
        image = self._input_preprocessing(x_test, w,h)
        predictions = self._predict(self.gender_model, image, self.batch_size)
        idx = np.argmax(predictions)
        gender = self.gender_filter[idx]

        # Age Prediction
        w, h = self.age_target_size[0], self.age_target_size[1]
        image = self._input_preprocessing(x_test, w,h)
        predictions = self._predict(self.age_model,  image, self.batch_size)
        idx = np.argmax(predictions)
        age = self.age_filter[idx]

        return gender, age

## lazy-loading models
_classifier = None

def classify_demographics(image):
    global _classifier
    if not _classifier:
        _classifier = Demographics()
    return _classifier.classify(image)