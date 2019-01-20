

from huasca import cellar
import pathlib
from huasca import __file__ as _pkgrootfile
_rootpath = pathlib.PurePath(_pkgrootfile).parent


## True Negative
assert not cellar._exists('non/existent/file') , 'Can\'t tell when a file doesn\'t exist.'

## True Positive
assert cellar._exists(str(_rootpath.joinpath('bin/Arial.ttf'))) , 'Can\'t tell when a file exists.'

