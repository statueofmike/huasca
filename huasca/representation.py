
from PIL import Image as _Image
import numpy as _np

class Detection:
    """ Convenience object to contain detected object info. 
       Boxes follow PIL format of (left, upper, right, lower) 
       To be used with faces and generic objects. """
    
    @property
    def boxes(self):
        return self._boxes
    
    @property
    def scores(self):
        return self._scores

    @property
    def labels(self):
        return self._labels
    
    @property
    def base_image(self):
        return self._base_image

    @property
    def portraits(self):
        return [self.base_image.crop(box) for box in self.boxes]

    def __init__(self,boxes,scores,labels,base_image):
        self._boxes = boxes
        self._scores = scores

        if isinstance(base_image,_np.ndarray):
            base_image = _Image.fromarray(base_image)
        self._base_image = base_image

    # TODO add annotation/labelling
