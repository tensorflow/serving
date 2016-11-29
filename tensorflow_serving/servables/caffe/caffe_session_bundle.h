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


#ifndef TENSORFLOW_SERVING_SERVABLES_CAFFE_SESSION_BUNDLE_H_
#define TENSORFLOW_SERVING_SERVABLES_CAFFE_SESSION_BUNDLE_H_

#include <memory>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow_serving/servables/caffe/caffe_serving_session.h"

#include "tensorflow/core/framework/tensor.pb.h"

namespace tensorflow {
namespace serving {

// Low-level functionality for setting up a Caffe inference Session.
const char kGraphDefFilename[] = "deploy.prototxt";
const char kVariablesFilename[] = "weights.caffemodel";
const char kGraphTxtLabelFilename[] = "classlabels.txt";

// A global initialization function that you should call in your main function.
// Currently it just invokes caffe::GlobalInit(..)
void CaffeGlobalInit(int* pargc, char*** pargv);

// Remarks: Very roughly equivalent to a TF session bundle
struct CaffeSessionBundle {
  std::unique_ptr<tensorflow::Session> session;
  CaffeMetaGraphDef meta_graph_def;
};

// Loads a manifest and initialized session using the output of an Exporter
tensorflow::Status LoadSessionBundleFromPath(
    const CaffeSessionOptions& options,
    const tensorflow::StringPiece export_dir, CaffeSessionBundle* bundle);

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_CAFFE_SESSION_BUNDLE_H_