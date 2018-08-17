from PIL import Image
import os

import huasca
_image = Image.open(os.path.abspath(os.path.dirname(__file__))+'/images/tvmonitor.png')
annotated,classes,scores,boxes = huasca.detection.ObjectDetector().detect(_image)

assert 'tvmonitor' in classes , "FAIL - ObjectDetector didn't detect 'tvmonitor'."
print("PASS - object detection")