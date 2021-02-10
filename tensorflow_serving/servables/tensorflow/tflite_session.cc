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

#include <memory>
#include <string>
#include <utility>

#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/hashtable/hashtable_ops.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/kernels/parse_example/parse_example.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/tools/signature/signature_def_util.h"
#include "tensorflow/lite/util.h"
#include "tensorflow_serving/servables/tensorflow/tflite_interpreter_pool.h"

namespace tensorflow {
namespace serving {

// Map of TFLite tensor name to <TF TensorInfo, TFLite tensor index>.
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

Status GetTensorInfoMap(const tflite::Interpreter* interpreter, bool input,
                        TensorInfoMap* infomap) {
  const std::vector<int>& indices =
      input ? interpreter->inputs() : interpreter->outputs();
  const string& input_str = input ? "Input" : "Output";
  for (int index : indices) {
    const TfLiteTensor* tensor = interpreter->tensor(index);
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

std::vector<int> TensorDims(const Tensor* tensor) {
  std::vector<int> dims(tensor->dims());
  for (int i = 0; i < tensor->dims(); ++i) {
    dims[i] = static_cast<int>(tensor->dim_size(i));
  }
  return dims;
}

// Create output tensors making sure they are the right size. //
Status CreateOutputTensors(
    std::unique_ptr<internal::TfLiteInterpreterPool>& interpreter_pool,
    const std::vector<string>& output_tensor_names,
    const std::map<string, int>& output_tensor_to_idx,
    std::map<int32_t, Tensor*>& tflite_idx_to_output_tensor,
    std::vector<Tensor>* output_tensors, bool use_batch_parallelism = false,
    int actual_batch_size = 0) {
  output_tensors->reserve(output_tensor_names.size());
  for (std::string tfname : output_tensor_names) {
    auto fix_status = FixTfLiteTensorName(output_tensor_to_idx, tfname);
    if (fix_status != Status::OK()) {
      return errors::Internal("Missing output TFLite tensor: ", tfname, ": ",
                              fix_status.error_message());
    }
    const int tflite_idx = output_tensor_to_idx.at(tfname);
    TensorShape tf_shape;
    const auto& interpreter = interpreter_pool->GetInterpreter(0)->Get();
    const auto* tflite_tensor = interpreter->tensor(tflite_idx);
    if (use_batch_parallelism) {
      tf_shape.AddDim(actual_batch_size);
    } else if (tflite_tensor->dims->size > 0) {
      tf_shape.AddDim(tflite_tensor->dims->data[0]);
    }
    for (int i = 1; i < tflite_tensor->dims->size; ++i) {
      tf_shape.AddDim(tflite_tensor->dims->data[i]);
    }
    DataType tf_type;
    TF_RETURN_IF_ERROR(TfLiteTypeToTfType(tflite_tensor->type, &tf_type));
    output_tensors->emplace_back(tf_type, tf_shape);
    tflite_idx_to_output_tensor[tflite_idx] = &output_tensors->back();
  }
  return Status::OK();
}

Status SetInputAndInvokeMiniBatch(
    std::unique_ptr<internal::TfLiteInterpreterPool>& interpreter_pool,
    const std::map<int, const Tensor*>& tflite_idx_to_tf_input_tensor,
    bool use_batch_parallelism = false, int32_t num_minibatches = 1,
    int minibatch = 0) {
  const auto& interpreter_wrapper = interpreter_pool->GetInterpreter(minibatch);
  auto* interpreter = interpreter_wrapper->Get();
  // Load input data from Tensorflow tensors.
  for (const auto entry : tflite_idx_to_tf_input_tensor) {
    int tflite_input_idx = entry.first;
    auto tflite_input_tensor = interpreter->tensor(tflite_input_idx);
    const auto* tf_input_tensor = entry.second;
    if (tflite_input_tensor->type != kTfLiteString) {
      if (use_batch_parallelism) {
        return errors::Internal(
            "Batch parallelism should not be enabled for non-string inputs");
      }
      auto tensor_bytes = tf_input_tensor->tensor_data();
      if (tensor_bytes.size() != tflite_input_tensor->bytes) {
        if (interpreter->ResizeInputTensor(
                tflite_input_idx, TensorDims(tf_input_tensor)) != kTfLiteOk) {
          return errors::Internal(
              "Failed to resize input tensor: ", tflite_input_tensor->name,
              " from ", tflite_input_tensor->bytes, " to ", tensor_bytes.size(),
              " bytes.");
        }
        if (interpreter->AllocateTensors() != kTfLiteOk) {
          return errors::Internal("Failed to allocate tensors");
        }
      }
      std::memcpy(tflite_input_tensor->data.raw, tensor_bytes.data(),
                  tensor_bytes.size());
    } else {
      // Copy the string tensor data to the input tflite tensor.
      const int fixed_batch_size = interpreter_pool->FixedBatchSize();
      const int actual_batch_size = entry.second->NumElements();
      const int begin = fixed_batch_size * minibatch;
      int end = static_cast<int>(actual_batch_size);
      if (use_batch_parallelism) {
        end = std::min(end, fixed_batch_size * (minibatch + 1));
      }
      const auto batch = gtl::ArraySlice<tstring>(entry.second->flat<tstring>())
                             .subspan(begin, end - begin);
      const int minibatch_size = batch.size();
      // Always resize when not using parallelism.
      const bool needs_resize =
          use_batch_parallelism
              ? minibatch_size > interpreter_wrapper->GetMiniBatchSize()
              : minibatch_size != interpreter_wrapper->GetMiniBatchSize();
      if (needs_resize) {
        interpreter->ResizeInputTensor(entry.first, {minibatch_size});
        interpreter_wrapper->SetMiniBatchSize(minibatch_size);
        if (interpreter->AllocateTensors() != kTfLiteOk) {
          return errors::Internal("Failed to allocate tensors");
        }
      }
      TF_RETURN_IF_ERROR(interpreter_wrapper->SetStringData(
          batch, tflite_input_tensor, tflite_input_idx));
    }
  }
  if (interpreter_wrapper->Invoke() != kTfLiteOk) {
    return errors::Internal("Failed to invoke TfLite interpreter");
  }
  return Status::OK();
}

Status SetMiniBatchOutput(
    std::unique_ptr<internal::TfLiteInterpreterPool>& interpreter_pool,
    const std::map<int, Tensor*>& tflite_idx_to_output_tensor,
    std::vector<Tensor>* outputs, bool use_batch_parallelism = false,
    int actual_batch_size = 0, int num_minibatches = 1, size_t minibatch = 0) {
  const int fixed_batch_size = interpreter_pool->FixedBatchSize();
  for (const auto& entry : tflite_idx_to_output_tensor) {
    Tensor* tensor = entry.second;
    const DataType tf_type = tensor->dtype();
    const int begin = fixed_batch_size * minibatch;
    const int end = std::min(fixed_batch_size * static_cast<int>(minibatch + 1),
                             actual_batch_size);
    const int actual_minibatch_size = end - begin;
    const auto& interpreter_wrapper =
        interpreter_pool->GetInterpreter(minibatch);
    const auto* interpreter = interpreter_wrapper->Get();
    auto tflite_tensor = interpreter->tensor(entry.first);
    if (DataTypeCanUseMemcpy(tf_type)) {
      auto tensor_bytes = tensor->tensor_data();
      int offset = 0;
      size_t tflite_tensor_bytes = tflite_tensor->bytes;
      if (use_batch_parallelism) {
        // Each batch will produce minibatch_size tensors.
        // We want to make sure we only copy actual_batch_size.
        offset = minibatch * tflite_tensor_bytes;
        tflite_tensor_bytes =
            tflite_tensor_bytes / fixed_batch_size * actual_minibatch_size;
      }
      std::memcpy(const_cast<char*>(tensor_bytes.data() + offset),
                  tflite_tensor->data.raw, tflite_tensor_bytes);
    } else if (tflite_tensor->type == kTfLiteString) {
      const int string_count = tflite::GetStringCount(tflite_tensor);
      int num_strings = string_count;
      int offset = 0;
      if (use_batch_parallelism) {
        offset = minibatch * string_count;
        num_strings = string_count / fixed_batch_size * actual_minibatch_size;
      }
      auto str_tensors = tensor->flat<tstring>();
      for (int i = 0; i < num_strings; i++) {
        const auto& ref = tflite::GetString(tflite_tensor, i);
        str_tensors(i + offset).assign(ref.str, ref.len);
      }
    }
  }
  return Status::OK();
}

typedef std::function<Status(std::unique_ptr<internal::TfLiteInterpreterPool>&,
                             const std::map<int, const Tensor*>&, bool, int32_t,
                             int)>
    BatchFunction;
void ParallelFor(
    const BatchFunction& f,
    std::unique_ptr<internal::TfLiteInterpreterPool>& interpreter_pool,
    const std::map<int, const Tensor*>& tflite_inputs, int n,
    bool use_batch_parallelism, bool run_in_caller,
    std::vector<Status>& status) {
  if (n == 0) return;
  auto* thread_pool = interpreter_pool->ThreadPool();
  if (thread_pool == nullptr) {
    for (int i = 0; i < n; ++i) {
      status[i] =
          f(interpreter_pool, tflite_inputs, use_batch_parallelism, n, i);
    }
    return;
  }
  int num_jobs = run_in_caller ? n - 1 : n;
  int first_job = run_in_caller ? 1 : 0;
  tensorflow::BlockingCounter counter(num_jobs);
  for (int i = first_job; i < n; ++i) {
    thread_pool->Schedule([&f, &interpreter_pool, &tflite_inputs,
                           use_batch_parallelism, n, i, &counter, &status] {
      status[i] =
          f(interpreter_pool, tflite_inputs, use_batch_parallelism, n, i);
      counter.DecrementCount();
    });
  }
  if (run_in_caller) {
    status[0] = f(interpreter_pool, tflite_inputs, use_batch_parallelism, n, 0);
  }
  counter.Wait();
}

Status RunBatchParallel(
    std::unique_ptr<internal::TfLiteInterpreterPool>& interpreter_pool,
    const std::vector<std::pair<string, Tensor>>& inputs,
    const std::map<int, const Tensor*>& tflite_idx_to_input_tensor,
    bool run_in_caller_thread, const std::vector<string>& output_tensor_names,
    const std::map<string, int>& output_tensor_to_index,
    std::vector<Tensor>* outputs) {
  const int actual_batch_size = inputs[0].second.NumElements();
  const int32_t num_threads = interpreter_pool->NumInterpreters();
  const int min_batch_size = interpreter_pool->FixedBatchSize();
  int num_minibatches = std::min(
      num_threads, (actual_batch_size + min_batch_size - 1) / min_batch_size);
  std::vector<Status> status_of_minibatches(num_minibatches);
  ParallelFor(SetInputAndInvokeMiniBatch, interpreter_pool,
              tflite_idx_to_input_tensor, num_minibatches, true,
              run_in_caller_thread, status_of_minibatches);
  for (Status& status_of_minibatch : status_of_minibatches) {
    TF_RETURN_IF_ERROR(status_of_minibatch);
  }
  std::map<int32_t, Tensor*> tflite_idx_to_output_tensor;
  TF_RETURN_IF_ERROR(CreateOutputTensors(
      interpreter_pool, output_tensor_names, output_tensor_to_index,
      tflite_idx_to_output_tensor, outputs, true, actual_batch_size));

  // Set the contents of the return tensors.
  for (size_t i = 0; i < num_minibatches; ++i) {
    TF_RETURN_IF_ERROR(SetMiniBatchOutput(
        interpreter_pool, tflite_idx_to_output_tensor, outputs, true,
        actual_batch_size, num_minibatches, i));
  }
  return Status::OK();
}

}  // namespace

Status TfLiteSession::Create(string&& buffer, const SessionOptions& options,
                             int num_pools, int num_interpreters_per_pool,
                             std::unique_ptr<TfLiteSession>* tflite_session,
                             ::google::protobuf::Map<string, SignatureDef>* signatures,
                             bool run_in_caller_thread) {
  auto model = tflite::FlatBufferModel::BuildFromModel(
      flatbuffers::GetRoot<tflite::Model>(buffer.data()));
  if (model == nullptr) {
    return errors::InvalidArgument("Cannot build FlatBufferModel from buffer.");
  }

  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::ops::custom::AddParseExampleOp(&resolver);
  tflite::ops::custom::AddHashtableOps(&resolver);

  std::unique_ptr<tflite::Interpreter> interpreter;
  if (tflite::InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk) {
    return errors::Internal("Cannot build Interpreter from buffer.");
  }

  TensorInfoMap inputs;
  TF_RETURN_IF_ERROR(GetTensorInfoMap(interpreter.get(), true, &inputs));
  TensorInfoMap outputs;
  TF_RETURN_IF_ERROR(GetTensorInfoMap(interpreter.get(), false, &outputs));

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

  num_pools = std::max(1, num_pools);
  num_interpreters_per_pool = std::max(1, num_interpreters_per_pool);

  std::unique_ptr<internal::TfLiteSessionPool> session_pool;
  TF_RETURN_IF_ERROR(internal::TfLiteSessionPool::CreateTfLiteSessionPool(
      model.get(), options, run_in_caller_thread, num_pools,
      num_interpreters_per_pool, session_pool));

  tflite_session->reset(new TfLiteSession(
      std::move(input_tensor_to_index), std::move(output_tensor_to_index),
      std::move(buffer), std::move(model), std::move(session_pool),
      run_in_caller_thread));
  return Status::OK();
}

TfLiteSession::TfLiteSession(
    std::map<string, int>&& input_tensor_to_index,
    std::map<string, int>&& output_tensor_to_index, string&& buffer,
    std::unique_ptr<tflite::FlatBufferModel> model,
    std::unique_ptr<internal::TfLiteSessionPool> session_pool,
    bool run_in_caller_thread)
    : input_tensor_to_index_(std::move(input_tensor_to_index)),
      output_tensor_to_index_(std::move(output_tensor_to_index)),
      model_serialized_bytes_(std::move(buffer)),
      model_(std::move(model)),
      session_pool_(std::move(session_pool)),
      run_in_caller_thread_(run_in_caller_thread) {}

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
  std::map<int, const Tensor*> tflite_idx_to_input_tensor;
  for (const auto& input : inputs) {
    string name = input.first;
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        FixTfLiteTensorName(input_tensor_to_index_, name),
        "Missing input TFLite tensor: ", name);
    const int index = input_tensor_to_index_.at(name);
    tflite_idx_to_input_tensor[index] = &input.second;
  }

#define RETURN_POOL_IF_ERROR(...)                                        \
  do {                                                                   \
    ::tensorflow::Status _status = (__VA_ARGS__);                        \
    if (TF_PREDICT_FALSE(!_status.ok())) {                               \
      session_pool_->ReturnInterpreterPool(std::move(interpreter_pool)); \
      return _status;                                                    \
    }                                                                    \
  } while (0);
  auto interpreter_pool = session_pool_->GetInterpreterPool();
  if (interpreter_pool->UseBatchParallelism()) {
    RETURN_POOL_IF_ERROR(
        RunBatchParallel(interpreter_pool, inputs, tflite_idx_to_input_tensor,
                         run_in_caller_thread_, output_tensor_names,
                         output_tensor_to_index_, outputs));
    session_pool_->ReturnInterpreterPool(std::move(interpreter_pool));
    return Status::OK();
  }

  RETURN_POOL_IF_ERROR(
      SetInputAndInvokeMiniBatch(interpreter_pool, tflite_idx_to_input_tensor));

  // Create return tensors and map the tflite tensor index to the
  // index of the created tensor.
  std::map<int32_t, Tensor*> tflite_idx_to_output_tensor;
  RETURN_POOL_IF_ERROR(CreateOutputTensors(
      interpreter_pool, output_tensor_names, output_tensor_to_index_,
      tflite_idx_to_output_tensor, outputs));

  // Set the contents of the return tensors.
  RETURN_POOL_IF_ERROR(SetMiniBatchOutput(
      interpreter_pool, tflite_idx_to_output_tensor, outputs));

#undef RETURN_POOL_IF_ERROR
  session_pool_->ReturnInterpreterPool(std::move(interpreter_pool));
  return Status::OK();
}

Status TfLiteSession::ListDevices(std::vector<DeviceAttributes>* response) {
  return errors::Unimplemented("ListDevices is not yet supported.");
}

}  // namespace serving
}  // namespace tensorflow
