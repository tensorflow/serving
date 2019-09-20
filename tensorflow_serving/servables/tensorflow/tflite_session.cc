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

#include "tensorflow_serving/servables/tensorflow/tflite_session.h"

#include <string>
#include <utility>

#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"

namespace tensorflow {
namespace serving {

// Map of TFLite tensor name to <TF TensorInfo, TFLite tensor index>.
using TensorInfoMap = std::map<string, std::pair<TensorInfo, int>>;

namespace {

Status TfLiteTypeToTfType(TfLiteType tflite_type, DataType* type) {
  switch (tflite_type) {
    case kTfLiteNoType:
      *type = tensorflow::DT_INVALID;
      break;
    case kTfLiteFloat32:
      *type = tensorflow::DT_FLOAT;
      break;
    case kTfLiteInt32:
      *type = tensorflow::DT_INT32;
      break;
    case kTfLiteUInt8:
      *type = tensorflow::DT_UINT8;
      break;
    case kTfLiteInt64:
      *type = tensorflow::DT_INT64;
      break;
    case kTfLiteString:
      *type = tensorflow::DT_STRING;
      break;
    case kTfLiteBool:
      *type = tensorflow::DT_BOOL;
      break;
    case kTfLiteInt16:
      *type = tensorflow::DT_INT16;
      break;
    case kTfLiteComplex64:
      *type = tensorflow::DT_COMPLEX64;
      break;
    case kTfLiteInt8:
      *type = tensorflow::DT_INT8;
      break;
    default:
      return errors::Internal("Unknown TfLite type: ", tflite_type);
  }
  return Status::OK();
}

std::string TfLiteToTensorName(const string& name) {
  // TF variable names have ':0' suffix, TF Lite variables dont.
  std::pair<absl::string_view, absl::string_view> name_index =
      absl::StrSplit(name, absl::MaxSplits(':', 1));
  return std::string(name_index.first);
}

Status TfLiteTensorToTensorInfo(const TfLiteTensor* tflite_tensor,
                                TensorInfo* info) {
  DataType tf_type;
  TF_RETURN_IF_ERROR(TfLiteTypeToTfType(tflite_tensor->type, &tf_type));
  info->set_dtype(tf_type);
  info->set_name(tflite_tensor->name);
  for (int i = 0; i < tflite_tensor->dims->size; i++) {
    info->mutable_tensor_shape()->add_dim()->set_size(
        tflite_tensor->dims->data[i]);
  }
  return Status::OK();
}

Status GetTensorInfoMap(const tflite::Interpreter& interpreter, bool input,
                        TensorInfoMap* infomap) {
  const std::vector<int>& indices =
      input ? interpreter.inputs() : interpreter.outputs();
  const string& input_str = input ? "Input" : "Output";
  for (int index : indices) {
    const TfLiteTensor* tensor = interpreter.tensor(index);
    if (tensor->name == nullptr) {
      return errors::Internal(input_str,
                              " name missing for tensor index: ", index);
    }
    TensorInfo info;
    TF_RETURN_IF_ERROR(TfLiteTensorToTensorInfo(tensor, &info));
    if (!infomap->emplace(tensor->name, std::pair<TensorInfo, int>(info, index))
             .second) {
      return errors::AlreadyExists(input_str, " tensor name: ", tensor->name,
                                   " has multiple indices");
    }
  }
  return Status::OK();
}

std::vector<int> TensorDims(const Tensor& tensor) {
  std::vector<int> dims;
  dims.reserve(tensor.dims());
  for (int i = 0; i < tensor.dims(); ++i) {
    dims.push_back(static_cast<int>(tensor.dim_size(i)));
  }
  return dims;
}

Status FillTfLiteTensorFromInput(const string& name, const Tensor& tensor,
                                 tflite::Interpreter* interpreter, int index) {
  auto tflite_tensor = interpreter->tensor(index);
  if (tflite_tensor == nullptr) {
    return errors::InvalidArgument("Failed to get TFLite tensor: ", name,
                                   " at index: ", index);
  }
  if (tflite_tensor->type != kTfLiteString) {
    auto tensor_bytes = tensor.tensor_data();
    if (tensor_bytes.size() != tflite_tensor->bytes) {
      // Slow path: needs resize+realloc.
      // TODO(b/140959776): Reduce the chance of taking this path by either
      // having multiple instances of interpreter or sub-graphs for commonly
      // used input sizes.
      if (interpreter->ResizeInputTensor(index, TensorDims(tensor)) !=
          kTfLiteOk) {
        return errors::Internal("Failed to resize input tensor: ", name,
                                " from ", tflite_tensor->bytes, " to ",
                                tensor_bytes.size(), " bytes.");
      }
      if (interpreter->AllocateTensors() != kTfLiteOk) {
        return errors::Internal(
            "Failed to AllocateTensors() due to change in input tensor ", name,
            " size from ", tflite_tensor->bytes, " to ", tensor_bytes.size(),
            " bytes.");
      }
    }
    std::memcpy(tflite_tensor->data.raw, tensor_bytes.data(),
                tensor_bytes.size());
  } else {
    tflite::DynamicBuffer buf;
    auto tensor_vec = tensor.vec<tstring>();
    for (int i = 0; i < tensor_vec.size(); i++) {
      buf.AddString(tensor_vec(i).data(), tensor_vec(i).size());
    }
    buf.WriteToTensor(tflite_tensor, /*new_shape=*/nullptr);
  }
  return Status::OK();
}

}  // namespace

Status TfLiteSession::Create(string&& buffer,
                             std::unique_ptr<TfLiteSession>* tflite_session,
                             ::google::protobuf::Map<string, SignatureDef>* signatures) {
  auto model = tflite::FlatBufferModel::BuildFromModel(
      flatbuffers::GetRoot<tflite::Model>(buffer.data()));
  if (model == nullptr) {
    return errors::InvalidArgument("Cannot build FlatBufferModel from buffer.");
  }

  // TODO(b/140959776): Add support for non-builtin ops (flex or custom ops).
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  if (tflite::InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk) {
    return errors::Internal("Cannot build Interpreter from buffer.");
  }
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    return errors::Internal("Cannot allocator tensors in Interpreter.");
  }

  TensorInfoMap inputs;
  TF_RETURN_IF_ERROR(GetTensorInfoMap(*interpreter, true, &inputs));
  TensorInfoMap outputs;
  TF_RETURN_IF_ERROR(GetTensorInfoMap(*interpreter, false, &outputs));

  std::map<string, int> input_tensor_to_index;
  std::map<string, int> output_tensor_to_index;

  // Build a default SignatureDef map.
  // TODO(b/140959776): Add support to read this map from tflite model.
  signatures->clear();
  SignatureDef* sigdef = &(*signatures)[kDefaultServingSignatureDefKey];
  for (const auto& info : inputs) {
    (*sigdef->mutable_inputs())[info.first] = info.second.first;
    input_tensor_to_index[info.first] = info.second.second;
  }
  for (const auto& info : outputs) {
    (*sigdef->mutable_outputs())[info.first] = info.second.first;
    output_tensor_to_index[info.first] = info.second.second;
  }
  sigdef->set_method_name(kPredictMethodName);

  tflite_session->reset(new TfLiteSession(
      std::move(input_tensor_to_index), std::move(output_tensor_to_index),
      std::move(buffer), std::move(model), std::move(interpreter)));
  return Status::OK();
}

TfLiteSession::TfLiteSession(std::map<string, int>&& input_tensor_to_index,
                             std::map<string, int>&& output_tensor_to_index,
                             string&& buffer,
                             std::unique_ptr<tflite::FlatBufferModel> model,
                             std::unique_ptr<tflite::Interpreter> interpreter)
    : input_tensor_to_index_(std::move(input_tensor_to_index)),
      output_tensor_to_index_(std::move(output_tensor_to_index)),
      model_serialized_bytes_(std::move(buffer)),
      model_(std::move(model)),
      interpreter_(std::move(interpreter)) {}

Status TfLiteSession::Run(const std::vector<std::pair<string, Tensor>>& inputs,
                          const std::vector<string>& output_tensor_names,
                          const std::vector<string>& target_node_names,
                          std::vector<Tensor>* outputs) {
  RunMetadata run_metadata;
  return Run(RunOptions(), inputs, output_tensor_names, target_node_names,
             outputs, &run_metadata);
}

Status TfLiteSession::Run(const RunOptions& run_options,
                          const std::vector<std::pair<string, Tensor>>& inputs,
                          const std::vector<string>& output_tensor_names,
                          const std::vector<string>& target_node_names,
                          std::vector<Tensor>* outputs,
                          RunMetadata* run_metadata) {
  // TODO(b/140959776): Remove serialized Run() calls, and support
  // multi-threaded execution -- allowing multiple Run() calls to
  // happen in-parallel.
  absl::MutexLock lock(&mutex_);
  for (const auto& input : inputs) {
    const string& name = TfLiteToTensorName(input.first);
    if (input_tensor_to_index_.find(name) == input_tensor_to_index_.end()) {
      return errors::InvalidArgument("Missing input TFLite tensor: ", name);
    }
    const int index = input_tensor_to_index_.at(name);
    TF_RETURN_IF_ERROR(FillTfLiteTensorFromInput(name, input.second,
                                                 interpreter_.get(), index));
  }

  if (interpreter_->Invoke() != kTfLiteOk) {
    return errors::Internal("Failed to run interpreter.");
  }

  outputs->clear();
  for (const auto& tfname : output_tensor_names) {
    const string& name = TfLiteToTensorName(tfname);
    if (output_tensor_to_index_.find(name) == output_tensor_to_index_.end()) {
      return errors::InvalidArgument("Missing output TFLite tensor: ", name);
    }
    const int index = output_tensor_to_index_.at(name);
    auto* tflite_tensor = interpreter_->tensor(index);
    if (tflite_tensor == nullptr) {
      return errors::InvalidArgument(
          "Failed to get output TFLite tensor: ", name, " at index: ", index);
    }

    DataType tf_type;
    TF_RETURN_IF_ERROR(TfLiteTypeToTfType(tflite_tensor->type, &tf_type));

    TensorShape shape;
    for (int i = 0; i < tflite_tensor->dims->size; i++) {
      shape.AddDim(tflite_tensor->dims->data[i]);
    }
    outputs->emplace_back(Tensor(tf_type, shape));
    auto tensor_bytes = outputs->back().tensor_data();
    if (tflite_tensor->bytes != tensor_bytes.size()) {
      return errors::Internal(
          "Failed to convert output tensor: ", name,
          " to TFLite tensor. Size mismatch: ", tensor_bytes.size(), " vs ",
          tflite_tensor->bytes);
    }
    std::memcpy(const_cast<char*>(tensor_bytes.data()), tflite_tensor->data.raw,
                tflite_tensor->bytes);
  }
  return Status::OK();
}

Status TfLiteSession::ListDevices(std::vector<DeviceAttributes>* response) {
  return errors::Unimplemented("ListDevices is not yet supported.");
}

}  // namespace serving
}  // namespace tensorflow
