from PIL import Image
import huasca
import os

obj = huasca.detection.TinyYolo()

_image = Image.open(os.path.abspath(os.path.dirname(__file__))+'/images/tvmonitor.png')

annotated,classes,scores = obj.process(_image)

assert('tvmonitor' in classes,"Tiny Yolo didn't detect 'tvmonitor'.")

print("tiny yolo test done")