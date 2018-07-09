import os as _os

from . import detection

with open(_os.path.abspath(_os.path.dirname(__file__))+'/__doc__','r') as f:
    __doc__ = f.read()
