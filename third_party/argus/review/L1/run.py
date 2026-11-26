"""Canonical L1 entry; preserve the existing review implementation."""
from pathlib import Path
import runpy
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if __name__ == '__main__':
    runpy.run_module('review', run_name='__main__')
