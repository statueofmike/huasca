"""
    run all scripts in this folder 
    Each one should try to just output "PASS - ..." or "FAIL ..."
"""

from os import listdir
from os import path
from importlib import import_module

workdir = path.abspath(path.dirname(__file__))
modules = [path.splitext(filename)[0] for filename in listdir(workdir) if path.splitext(filename)[1]=='.py']
modules = list(set(modules)-set(['__init__','__main__']))

print("Testing modules: {}".format(','.join(modules)))

for module in modules:
    import_module(module)

