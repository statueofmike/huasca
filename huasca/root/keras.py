# Wrapper to import Keras/TF quietly:
# https://github.com/keras-team/keras/commit/83aaadaa9d69214880d20b1e2bd9715a6c37fbe6
import sys as _sys
import os as _os
_stderr = _sys.stderr
_sys.stderr = open(_os.devnull, 'w')

from keras import *

_sys.stderr = _stderr
# disables some tensorflow noise (but not all)
_os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# silences ALL warnings, helps with tensorflow noise again
import warnings as _warnings
_warnings.simplefilter("ignore")
