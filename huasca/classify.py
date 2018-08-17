from keras.models import load_model
import os
from PIL import Image
import numpy as np

class  GenderClassifier():
    gender_filter = {0:'male',1:'female'}


    def __init__(self):
        self.image = None
        self.model = None
        target_size = (100,100)
        self.image_w = target_size[0]
        self.image_h = target_size[1]
        self.__load_model()

    def __load_model(self):
      self.model = load_model('./bin/gender_1.h5')


    def __cleanup(self):
        del self.model


    def process(self, image):
        self.image = image
        resized =     self.image.resize( (self.image_w, self.image_h), Image.ANTIALIAS) 
        self.preprocessed = resized.reshape(1,self.image_w,self.image_h,3)
        self.predictions = self.model.predict(self.preprocessed, batch_size=1)
        idx = np.argmax(self.predictions)
        return  self.gender_filter[idx],self.predictions[idx]


