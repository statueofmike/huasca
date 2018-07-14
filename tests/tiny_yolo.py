from PIL import Image
import os

import huasca
_image = Image.open(os.path.abspath(os.path.dirname(__file__))+'/images/tvmonitor.png')
annotated,classes,scores = huasca.detection.TinyYolo().process(_image)

assert 'tvmonitor' in classes , "FAIL - Tiny Yolo didn't detect 'tvmonitor'."
print("PASS - tiny yolo")