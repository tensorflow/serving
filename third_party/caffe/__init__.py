# Copyright IBM Corp. All Rights Reserved.

'''
This caffe package __init__ replaces the default caffe package initialization,
and is based on the original: https://github.com/BVLC/caffe/blob/de8ac32a02f3e324b0495f1729bff2446d402c2c/python/caffe/__init__.py

There are two important differences:

  1. In this implementation, The '_caffe' module is embedded within the
     host process as a built-in module; relative-path imports in the
     original pycaffe  implementation will fail; the custom
     EmbeddedCaffeImporter corrects this behavior.

  2. This package only exposes a subset of pycaffe to get python layers
     working correctly. Other parts of pycaffe, including those that
     ultimately import multiprocessing, are not guaranteed to work. '''

import sys

class EmbeddedCaffeImporter(object):
  name = 'caffe._caffe'

  def find_module(self, fullname, path):
    if fullname == self.name:
      return self

    return None

  def load_module(self, fullname):
    return __import__('_caffe')

sys.meta_path.append(EmbeddedCaffeImporter())

from .pycaffe import Net, SGDSolver, NesterovSolver, AdaGradSolver, RMSPropSolver, AdaDeltaSolver, AdamSolver
from ._caffe import set_mode_cpu, set_mode_gpu, set_device, Layer, get_solver, layer_type_list, set_random_seed
from ._caffe import __version__
