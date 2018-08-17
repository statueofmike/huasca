from PIL import Image
import huasca
import os

obj = huasca.classify.GenderClassifier()

_image = Image.open(os.path.abspath(os.path.dirname(__file__))+'/images/cage1.jpg')

gender, score = obj.process(_image)

assert(gender == "male","Tiny Yolo didn't detect 'tvmonitor'.")

print("gender test done")