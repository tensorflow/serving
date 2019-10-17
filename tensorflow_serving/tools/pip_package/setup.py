# Copyright 2017 Google Inc. All Rights Reserved.
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
"""TensorFlow Serving Python API.

TensorFlow Serving is a flexible, high-performance serving system for machine
learning models, designed for production environments.TensorFlow Serving makes
it easy to deploy new algorithms and experiments, while keeping the same server
architecture and APIs. TensorFlow Serving provides out-of-the-box integration
with TensorFlow models, but can be easily extended to serve other types of
models and data.

This package contains the TensorFlow Serving Python APIs.
"""

import sys

from setuptools import find_packages
from setuptools import setup

DOCLINES = __doc__.split('\n')

# Set when releasing a new version of TensorFlow Serving (e.g. 1.0.0).
_VERSION = '1.15.0'
# Have this by default be open; releasing a new version will lock to TF version
_TF_VERSION = '~=1.15.0'
_TF_VERSION_SANITIZED = _TF_VERSION.replace('-', '')

project_name = 'tensorflow-serving-api'
# Set when building the pip package
if '--project_name' in sys.argv:
  project_name_idx = sys.argv.index('--project_name')
  project_name = sys.argv[project_name_idx + 1]
  sys.argv.remove('--project_name')
  sys.argv.pop(project_name_idx)

_TF_REQ = ['tensorflow'+_TF_VERSION_SANITIZED]

# GPU build (note: the only difference is we depend on tensorflow-gpu so
# pip doesn't overwrite it with the CPU build)
if 'tensorflow-serving-api-gpu' in project_name:
  _TF_REQ = ['tensorflow-gpu'+_TF_VERSION_SANITIZED]


REQUIRED_PACKAGES = [
    'grpcio>=1.0<2',
    'protobuf>=3.6.0',
] + _TF_REQ

setup(
    name=project_name,
    version=_VERSION.replace('-', ''),
    author='Google Inc.',
    author_email='tensorflow-serving-dev@googlegroups.com',
    packages=find_packages(),
    description=DOCLINES[0],
    long_description='\n'.join(DOCLINES[2:]),
    license='Apache 2.0',
    url='http://tensorflow.org/serving',
    keywords='tensorflow serving machine learning api libraries',
    install_requires=REQUIRED_PACKAGES,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
