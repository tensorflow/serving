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

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"

#ifdef WITH_PYTHON_LAYER
#include "python_prelude.h"
// embedded caffe python module initialization function
extern "C" void init_caffe();
#endif

#include "tensorflow_serving/servables/caffe/caffe_py_util.h"

namespace tensorflow {
namespace serving {

bool IsPyCaffeAvailable() {
#ifdef WITH_PYTHON_LAYER
  return true;
#else
  return false;
#endif
}

tensorflow::Status EnsurePyCaffe() {
  if (!IsPyCaffeAvailable()) {
    return tensorflow::errors::Internal(
        "Python unavilable in this build configuration");
  } else {
    return Status::OK();
  }
}

tensorflow::Status EnsurePyCaffeSystemPath(const string& path) {
  TF_RETURN_IF_ERROR(EnsurePyCaffe());
  TF_RETURN_IF_ERROR(EnsurePyCaffeInitialized());

#ifdef WITH_PYTHON_LAYER
  auto statement =
      strings::StrCat("import sys\nimport os\n", 
                      "if not '", path, "' in sys.path: ", 
                      "sys.path.insert(0, os.path.realpath('", path, "'))\n");

  PyRun_SimpleString(statement.c_str());
  return PythonStatus();
#endif
  return Status::OK();
}

tensorflow::Status EnsurePyCaffeInitialized() {
  static bool initialized = false;
  TF_RETURN_IF_ERROR(EnsurePyCaffe());

#ifdef WITH_PYTHON_LAYER
  if (!initialized) {
    LOG(INFO) << "Initializing Python:\n" << Py_GetVersion();
    // append the pythohn internal modules with py
    // the default module search path
    // make the caffe module accessible as a builtin
    if (PyImport_AppendInittab("_caffe", &init_caffe) == -1) {
      return tensorflow::errors::Internal(
          "Failed to add PyCaffe builtin module");
    }
    initialized = true;
  }
  if (!Py_IsInitialized()) {
    string path{Py_GetPath()};
    // causes a fatal error if initilization fails :(
    // also, make sure we dont prevent normal signal
    // handling (eg: SIGINT).
    Py_InitializeEx(0);
    // set sys.path to default search path.
    PySys_SetPath(path.c_str());
    // append site-specific paths to the module search path
    // and add other builtins.
    PyRun_SimpleString("import site;site.main()");
    if (PyErr_Occurred() != nullptr) {
      PyErr_PrintEx(0);
      return tensorflow::errors::Internal("Python initialization failed.");
    }
  }
#endif
  return Status::OK();
}

tensorflow::Status FinalizePyCaffe() {
  TF_RETURN_IF_ERROR(EnsurePyCaffe());
#ifdef WITH_PYTHON_LAYER
  Py_Finalize();
#endif
  return Status::OK();
}

tensorflow::Status PythonStatus() {
  TF_RETURN_IF_ERROR(EnsurePyCaffe());
#ifdef WITH_PYTHON_LAYER
  if (PyErr_Occurred() != nullptr) {
    // print error and clear error indicator
    PyErr_PrintEx(0);
    return tensorflow::errors::Internal("Python error occured.");
  }
#endif
  return Status::OK();
}

}  // namespace serving
}  // namespaces tensorflow
