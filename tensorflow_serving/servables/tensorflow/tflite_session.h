/* Copyright 2019 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_TFLITE_SESSION_H_
#define TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_TFLITE_SESSION_H_

#include <map>
#include <memory>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow_serving/servables/tensorflow/serving_session.h"

namespace tensorflow {
namespace serving {

// A session to run inference on a TensorFlow Lite model.
//
// EXPERIMENTAL: DO NOT use for production workloads.
class TfLiteSession : public ServingSession {
 public:
  // Creates a TfLiteSession object from `buffer` representing serialized
  // TFLite flatbuffer model. Also returns the SignatureDef map based on
  // input/outputs to the model.
  static Status Create(string&& buffer,
                       std::unique_ptr<TfLiteSession>* tflite_session,
                       ::google::protobuf::Map<string, SignatureDef>* signatures);

  ~TfLiteSession() override = default;

  Status Run(const std::vector<std::pair<string, Tensor>>& inputs,
             const std::vector<string>& output_tensor_names,
             const std::vector<string>& target_node_names,
             std::vector<Tensor>* outputs) override;

  Status Run(const RunOptions& run_options,
             const std::vector<std::pair<string, Tensor>>& inputs,
             const std::vector<string>& output_tensor_names,
             const std::vector<string>& target_node_names,
             std::vector<Tensor>* outputs, RunMetadata* run_metadata) override;

  Status ListDevices(std::vector<DeviceAttributes>* response) override;

 private:
  TfLiteSession(std::map<string, int>&& input_tensor_to_index,
                std::map<string, int>&& output_tensor_to_index, string&& buffer,
                std::unique_ptr<tflite::FlatBufferModel> model,
                std::unique_ptr<tflite::Interpreter> interpreter);

  const std::map<string, int> input_tensor_to_index_;
  const std::map<string, int> output_tensor_to_index_;
  const string model_serialized_bytes_;
  const std::unique_ptr<tflite::FlatBufferModel> model_;
  mutable absl::Mutex mutex_;
  std::unique_ptr<tflite::Interpreter> interpreter_ ABSL_GUARDED_BY(mutex_);

  TF_DISALLOW_COPY_AND_ASSIGN(TfLiteSession);
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_TFLITE_SESSION_H_
