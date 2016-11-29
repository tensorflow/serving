/*

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_SERVING_SERVABLES_CAFFE_PY_CAFFE_UTIL_H_
#define TENSORFLOW_SERVING_SERVABLES_CAFFE_PY_CAFFE_UTIL_H_

#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace serving {

// Returns true if Caffe was built with Python layer.
bool IsPyCaffeAvailable();

// Ensure python is loaded and initialized with the caffe
// wrapper module.
tensorflow::Status EnsurePyCaffeInitialized();

// frees all memory allocated by the Python interpreter. 
// errors during finalization are ignored.
tensorflow::Status FinalizePyCaffe();

// Ensure the given path is included in the python
// module search path (sys.path)
tensorflow::Status EnsurePyCaffeSystemPath(const string& path);

// returns an error if python has an error currently set
tensorflow::Status PythonStatus();

}  // namespace serving
}  // namespaces tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_CAFFE_PY_CAFFE_UTIL_H_