from PIL import Image
import huasca
import os

image = Image.open(os.path.abspath(os.path.dirname(__file__))+'/images/TC1.jpg')
gender,age = huasca.classify.demographics(image,verbose=False)

assert gender == "male","Gender Misclassified'."
assert age == '(38, 43)',"Age Misclassified"
