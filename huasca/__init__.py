import os as _os

from . import detection
from . import object_tracking

from .classify import GenderClassifier
del(classify)


with open(_os.path.abspath(_os.path.dirname(__file__))+'/__doc__','r') as _f:
    __doc__ = _f.read()
