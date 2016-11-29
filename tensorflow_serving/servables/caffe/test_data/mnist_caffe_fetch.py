# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/usr/bin/python2.7

"""Functions for downloading and extracting pretrained MNIST caffe models."""
from __future__ import print_function

import argparse
import tarfile
import os

from six.moves import urllib

VERSION_FORMAT_SPECIFIER = "%08d"
SOURCE_URL = 'https://ibm.box.com/shared/static/yemc4i8mtvrito2cgoypw4auxxvzatoo.tar'
OUT_FILE = 'mnist_pretrained_caffe.tar'
MODEL_FILES = ['classlabels.txt', 'deploy.prototxt', 'weights.caffemodel']

def maybe_download(url, filename, work_directory):
  """Download the data"""
  if not os.path.exists(work_directory):
    os.mkdir(work_directory)

  filepath = os.path.join(work_directory, filename)
  if not os.path.exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL, filepath)
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  
  return filepath

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("export_path", help="location to download and extract the model")
  parser.add_argument("--version", type=int, default=1, help="model version")
  args = parser.parse_args()

  export_dir = os.path.join(args.export_path, VERSION_FORMAT_SPECIFIER % args.version)
  if os.path.exists(export_dir):
    raise RuntimeError("Overwriting exports can cause corruption and are "
                       "not allowed. Duplicate export dir: %s" % export_dir)
  
  os.makedirs(export_dir)

  print('Downloading...', SOURCE_URL)
  filename = maybe_download(SOURCE_URL, OUT_FILE, export_dir)

  print('Extracting "%s" to "%s"' % (filename, export_dir))
  with tarfile.open(filename) as tar:
    tar.extractall(path=export_dir)

  for p in MODEL_FILES:
    if not os.path.isfile(os.path.join(export_dir, p)):
      raise FileNotFoundError("Expected model file '%s'" % p)
