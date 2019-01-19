
from PIL import Image as _Image
import numpy as _np
from PIL import ImageDraw, ImageFont  

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

    def _pad_box(self,box):
        """Pad a box size for context."""
        ratio = 0.2

        w = int(ratio*(box[2]-box[0]))
        h = int(ratio*(box[3]-box[1]))

        return (max(box[0]-w,0)
               ,max(box[1]-h,0)
               ,min(box[2]+w,self.base_image.size[0]) 
               ,min(box[3]+h,self.base_image.size[1])
                )


    # TODO add annotation/labelling  
    #image = Image.open("image.png")
    #draw  = ImageDraw.Draw(image)
    #font  = ImageFont.truetype("arial.ttf", 20, encoding="unic")
    #draw.text( (10,10), u"Your Text", fill=‘#a00000’, font=font)
    #image.save("out.png","PNG")
