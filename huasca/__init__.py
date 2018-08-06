import os as _os

from . import detection

from .object_tracking import tracking
del(object_tracking)

with open(_os.path.abspath(_os.path.dirname(__file__))+'/__doc__','r') as _f:
    __doc__ = _f.read()
