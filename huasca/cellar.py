
""" 
    ML Model Downloader
    Ensures presence of static-linked models.
    Possibly later expand to dynamic loading.

    e.g. str(_rootpath.joinpath('bin/yolov3-tiny_coco.h5'))
		`huasca.cellar.require(yolopath)`

"""

import requests
import os.path

import pathlib
from huasca import __file__ as _pkgrootfile
_rootpath = pathlib.PurePath(_pkgrootfile).parent

_cloudhost = 'https://github.com/solo-tiger/huasca/raw/obj/unpackaged/'

def require(resource_path):
	_download(resource_path)

def _exists(resource_path):
	return os.path.isfile(resource_path)

def _download(resource_path):
	if _exists(resource_path):
		return	

	filename = os.path.basename(resource_path)

	print("{} not present, downloading from Github...".format(filename))

	source = _cloudhost+filename
	destination = resource_path
	
	r = requests.get(source, allow_redirects=True)
	open(destination, 'wb').write(r.content)

	print("  ...Installed to {}.".format(filename,destination))
