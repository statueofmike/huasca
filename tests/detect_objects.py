
import huasca

import os
import pathlib
from PIL import Image

dog_image = Image.open(str(pathlib.PurePath(__file__).parent.joinpath('images/dog02.jpg')))
tvm_image = Image.open(str(pathlib.PurePath(__file__).parent.joinpath('images/tvmonitor.png')))

dets = huasca.detect.objects(dog_image,verbose=False)
assert 'dog' in dets.labels, 'Dog not identified correctly.'

dets = huasca.detect.objects(tvm_image,verbose=False)
assert 'tvmonitor' in dets.labels, 'Monitor not identified correctly.'

