"""
    run all scripts in this folder 
    Each one should try to just output "PASS - ..." or "FAIL ..."
"""

import sys
from os import listdir, path, getcwd,chdir
from importlib import import_module

workdir = path.abspath(path.dirname(__file__))
modules = [path.splitext(filename)[0] for filename in listdir(workdir) if path.splitext(filename)[1]=='.py']
modules = list(set(modules)-set(['__init__','__main__']))
modules.sort()

print("Testing modules [{}]:\n".format(','.join(modules)))

sys.path.append(path.abspath('../huasca'))

for module in modules:
    try:
        import_module(module)
        print("  PASS - module {}".format(module))
    except Exception as e:
        print(e)
        print("  FAIL - module {}".format(module))

