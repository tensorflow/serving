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

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

#if CUDA_VERSION >= 7050
#define EIGEN_HAS_CUDA_FP16
#endif

#include "tensorflow_serving/servables/caffe/caffe_session_bundle.h"

#include <string>
#include <utility>
#include <vector>
#include <memory>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/types.h"

#include "tensorflow_serving/servables/caffe/caffe_serving_session.h"

namespace tensorflow {
namespace serving {
namespace {

// Create a network using the given options and load the graph.
Status CreateSessionFromGraphDef(
    const CaffeSessionOptions& options, const CaffeMetaGraphDef& graph,
    std::unique_ptr<CaffeServingSession>* session) {
  session->reset(new CaffeServingSession(graph, options));
  return Status::OK();
}

Status GetClassLabelsFromExport(const StringPiece export_dir,
                                tensorflow::TensorProto* proto) {
  const string labels_path =
      tensorflow::io::JoinPath(export_dir, kGraphTxtLabelFilename);

  // default state
  proto->set_dtype(DT_INVALID);
  TensorShape({}).AsProto(proto->mutable_tensor_shape());

  if (Env::Default()->FileExists(labels_path).ok()) {
    /* Load labels. */
    std::unique_ptr<RandomAccessFile> file;
    TF_RETURN_IF_ERROR(Env::Default()->NewRandomAccessFile(labels_path, &file));
    {
      const size_t kBufferSizeBytes = 8192;
      io::InputBuffer in(file.get(), kBufferSizeBytes);

      string line;
      while (in.ReadLine(&line).ok()) {
        proto->add_string_val(line);
      }
    }
    proto->set_dtype(DT_STRING);
    TensorShape({1, proto->string_val().size()})
        .AsProto(proto->mutable_tensor_shape());
  }
  return Status::OK();
}

Status GetGraphDefFromExport(const StringPiece export_dir,
                             caffe::NetParameter* model_def) {
  const string model_def_path =
      tensorflow::io::JoinPath(export_dir, kGraphDefFilename);

  if (!Env::Default()->FileExists(model_def_path).ok()) {
    return errors::NotFound(
        strings::StrCat("Caffe model does not exist: ", model_def_path));
  } else if (!ReadProtoFromTextFile(model_def_path, model_def)) {
    return errors::InvalidArgument(strings::StrCat(
        "Caffe network failed to load from file: ", model_def_path));
  } else if (!UpgradeNetAsNeeded(model_def_path, model_def)) {
    return errors::InvalidArgument(
        strings::StrCat("Network upgrade failed from while loading from file: ",
                        model_def_path));
  }
  model_def->mutable_state()->set_phase(caffe::TEST);
  return Status::OK();
}

string GetVariablesFilename(const StringPiece export_dir) {
  return tensorflow::io::JoinPath(export_dir, kVariablesFilename);
}

Status RunRestoreOp(const StringPiece export_dir,
                    CaffeServingSession* session) {
  LOG(INFO) << "Running restore op for CaffeSessionBundle";
  string weights_path = GetVariablesFilename(export_dir);
  if (Env::Default()->FileExists(weights_path).ok()) {
    return session->CopyTrainedLayersFromBinaryProto(weights_path);
  } else {
    return errors::NotFound(
        strings::StrCat("Caffe weights file does not exist: ", weights_path));
  }
}

}  // namespace

tensorflow::Status LoadSessionBundleFromPath(const CaffeSessionOptions& options,
                                             const StringPiece export_dir,
                                             CaffeSessionBundle* bundle) {
  LOG(INFO) << "Attempting to load a SessionBundle from: " << export_dir;

  // load model prototxt
  TF_RETURN_IF_ERROR(
      GetGraphDefFromExport(export_dir, &(bundle->meta_graph_def.model_def)));

  // load class labels
  TF_RETURN_IF_ERROR(
      GetClassLabelsFromExport(export_dir, &(bundle->meta_graph_def.classes)));

  // resolve network inputs and outputs
  TF_RETURN_IF_ERROR(::caffe::ResolveNetInsOuts(
      bundle->meta_graph_def.model_def, bundle->meta_graph_def.resolved_inputs,
      bundle->meta_graph_def.resolved_outputs));
  // initialize network
  std::unique_ptr<CaffeServingSession> caffe_session;
  TF_RETURN_IF_ERROR(CreateSessionFromGraphDef(options, bundle->meta_graph_def,
                                               &caffe_session));

  // load weights
  TF_RETURN_IF_ERROR(RunRestoreOp(export_dir, caffe_session.get()));

  bundle->session.reset(caffe_session.release());

  LOG(INFO) << "Done loading SessionBundle";
  return Status::OK();
}

void CaffeGlobalInit(int* pargc, char*** pargv) {
  caffe::GlobalInit(pargc, pargv);
}

}  // namespace serving
}  // namespace tensorflow
