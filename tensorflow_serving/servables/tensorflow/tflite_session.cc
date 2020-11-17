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
#include "tensorflow/lite/kernels/hashtable/hashtable_ops.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/tools/signature/signature_def_util.h"
#include "tensorflow/lite/util.h"

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

std::string TfToTfLiteLegacyTensorName(const string& tf_name) {
  // TF variable names have ':0' suffix, early versions of the TF Lite converter
  // used to strip this suffix.
  std::pair<absl::string_view, absl::string_view> name_index =
      absl::StrSplit(tf_name, absl::MaxSplits(':', 1));
  return std::string(name_index.first);
}

// Checks that an input/output tensor actually exists. If not, attempts to
// update the tensor name with legacy TFLite tensor naming.
Status FixTfLiteTensorName(const std::map<string, int>& tensor_name_map,
                           string& tensor_name) {
  if (tensor_name_map.find(tensor_name) != tensor_name_map.end()) {
    return Status::OK();
  }

  // Try to update with the legacy tflite tensor name.
  const string& legacy_tflite_name = TfToTfLiteLegacyTensorName(tensor_name);
  if (tensor_name_map.find(legacy_tflite_name) != tensor_name_map.end()) {
    tensor_name = legacy_tflite_name;
    return Status::OK();
  }

  return errors::Internal("Unknown tensor '", tensor_name, "'.");
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
    auto tensor_vec = tensor.flat<tstring>();
    for (int i = 0; i < tensor_vec.size(); i++) {
      buf.AddString(tensor_vec(i).data(), tensor_vec(i).size());
    }
    TfLiteIntArray* dims_array = TfLiteIntArrayCreate(tensor.dims());
    for (int i = 0; i < tensor.dims(); i++) {
      dims_array->data[i] = static_cast<int>(tensor.dim_size(i));
    }
    // WriteToTensor() takes ownership of dims_array.
    buf.WriteToTensor(tflite_tensor, dims_array);
  }
  return Status::OK();
}

Status AppendTfLiteToTfTensorList(const TfLiteTensor* tflite_tensor,
                                  std::vector<Tensor>* outputs) {
  DataType tf_type;
  TF_RETURN_IF_ERROR(TfLiteTypeToTfType(tflite_tensor->type, &tf_type));

  TensorShape shape;
  for (int i = 0; i < tflite_tensor->dims->size; ++i) {
    shape.AddDim(tflite_tensor->dims->data[i]);
  }

  outputs->emplace_back(Tensor(tf_type, shape));
  Tensor* tensor = &outputs->back();
  if (DataTypeCanUseMemcpy(tf_type)) {
    auto tensor_bytes = tensor->tensor_data();
    if (tflite_tensor->bytes != tensor_bytes.size()) {
      return errors::Internal(
          "Failed to convert TFLite tensor: ", tflite_tensor->name,
          " to TF tensor. Size mismatch: ", tensor_bytes.size(), " vs ",
          tflite_tensor->bytes);
    }
    std::memcpy(const_cast<char*>(tensor_bytes.data()), tflite_tensor->data.raw,
                tflite_tensor->bytes);
  } else if (tflite_tensor->type == kTfLiteString) {
    const int num_strings = tflite::GetStringCount(tflite_tensor);
    if (num_strings != tensor->NumElements()) {
      return errors::Internal(
          "Failed to convert TFLite tensor: ", tflite_tensor->name,
          " to TF tensor. Num elements mismatch: ", tensor->NumElements(),
          " vs ", num_strings);
    }
    auto str_tensors = outputs->back().flat<tstring>();
    for (int i = 0; i < num_strings; i++) {
      auto ref = tflite::GetString(tflite_tensor, i);
      str_tensors(i).assign(ref.str, ref.len);
    }
  } else {
    return errors::Internal("TFLite to TF Tensor copy not supported for type: ",
                            tflite_tensor->type);
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
  // TODO(b/165643512): Remove adding Hashtable to resolver by default.
  tflite::ops::custom::AddHashtableOps(&resolver);
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

  // Map of TFLite tensor name -> tensor index
  std::map<string, int> input_tensor_to_index;
  std::map<string, int> output_tensor_to_index;
  for (const auto& info : inputs) {
    const string& tflite_tensor_name = info.first;
    input_tensor_to_index[tflite_tensor_name] = info.second.second;
  }
  for (const auto& info : outputs) {
    const string& tflite_tensor_name = info.first;
    output_tensor_to_index[tflite_tensor_name] = info.second.second;
  }

  // Attempt to read signature defs from the model file
  std::map<string, SignatureDef> signature_defs;
  const auto status =
      tflite::GetSignatureDefMap(model->GetModel(), &signature_defs);
  if (status != Status::OK()) {
    return errors::InvalidArgument(
        "Invalid SignatureDefs found in TfLite model: ",
        status.error_message());
  }
  const bool has_lite_signature_def = !signature_defs.empty();

  signatures->clear();
  if (has_lite_signature_def) {
    // Check that input/output tensors in the signature defs refer to existing
    // tensors.
    // If not found, try to match with legacy TFLite name (without suffix).
    for (const auto& signature_item : signature_defs) {
      SignatureDef* tflite_signature = &(*signatures)[signature_item.first];
      tflite_signature->CopyFrom(signature_item.second);
      for (auto& input : *tflite_signature->mutable_inputs()) {
        TensorInfo* tensor_info = &input.second;
        TF_RETURN_WITH_CONTEXT_IF_ERROR(
            FixTfLiteTensorName(input_tensor_to_index,
                                *tensor_info->mutable_name()),
            "Signature input ", input.first, " references an unknown tensor");
      }
      for (auto& output : *tflite_signature->mutable_outputs()) {
        TensorInfo* tensor_info = &output.second;
        TF_RETURN_WITH_CONTEXT_IF_ERROR(
            FixTfLiteTensorName(output_tensor_to_index,
                                *tensor_info->mutable_name()),
            "Signature output ", output.first, " references an unknown tensor");
      }
    }
  } else {
    // Build a mock signature from the input/output tensors of the model.
    // TODO(b/169239308)
    LOG(WARNING) << "No signature def found in TFLite model. Generating one.";
    SignatureDef* sigdef = &(*signatures)[kDefaultServingSignatureDefKey];
    for (const auto& info : inputs) {
      string tflite_tensor_name = TfToTfLiteLegacyTensorName(info.first);
      (*sigdef->mutable_inputs())[tflite_tensor_name] = info.second.first;
    }
    for (const auto& info : outputs) {
      string tflite_tensor_name = TfToTfLiteLegacyTensorName(info.first);
      (*sigdef->mutable_outputs())[tflite_tensor_name] = info.second.first;
    }
    sigdef->set_method_name(kPredictMethodName);
  }

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
  return Run(run_options, inputs, output_tensor_names, target_node_names,
             outputs, run_metadata, thread::ThreadPoolOptions());
}

Status TfLiteSession::Run(
    const RunOptions& run_options,
    const std::vector<std::pair<string, Tensor>>& inputs,
    const std::vector<string>& output_tensor_names,
    const std::vector<string>& target_node_names, std::vector<Tensor>* outputs,
    RunMetadata* run_metadata,
    const thread::ThreadPoolOptions& thread_pool_options) {
  // TODO(b/140959776): Remove serialized Run() calls, and support
  // multi-threaded execution -- allowing multiple Run() calls to
  // happen in-parallel.
  absl::MutexLock lock(&mutex_);
  for (const auto& input : inputs) {
    string name = input.first;
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        FixTfLiteTensorName(input_tensor_to_index_, name),
        "Missing input TFLite tensor: ", name);
    const int index = input_tensor_to_index_.at(name);
    TF_RETURN_IF_ERROR(FillTfLiteTensorFromInput(name, input.second,
                                                 interpreter_.get(), index));
  }

  if (interpreter_->Invoke() != kTfLiteOk) {
    return errors::Internal("Failed to run interpreter.");
  }

  outputs->clear();
  for (string name : output_tensor_names) {
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        FixTfLiteTensorName(output_tensor_to_index_, name),
        "Missing output TFLite tensor: ", name);
    const int index = output_tensor_to_index_.at(name);
    auto* tflite_tensor = interpreter_->tensor(index);
    if (tflite_tensor == nullptr) {
      return errors::InvalidArgument(
          "Failed to get output TFLite tensor: ", name, " at index: ", index);
    }
    TF_RETURN_IF_ERROR(AppendTfLiteToTfTensorList(tflite_tensor, outputs));
  }
  return Status::OK();
}

Status TfLiteSession::ListDevices(std::vector<DeviceAttributes>* response) {
  return errors::Unimplemented("ListDevices is not yet supported.");
}

}  // namespace serving
}  // namespace tensorflow
