import os, runpy, sys
sys.path.append(os.environ["TD_DEPS"])   # after site-packages: only fills in missing pure-python deps
script = sys.argv[1]; sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
runpy.run_path(script, run_name="__main__")
