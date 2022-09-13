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

#include "absl/functional/bind_front.h"
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/kernels/parse_example/parse_example.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/tools/signature/signature_def_util.h"
#include "tensorflow/lite/util.h"
#include "tensorflow_serving/batching/incremental_barrier.h"
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
  return OkStatus();
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
    return OkStatus();
  }

  // Try to update with the legacy tflite tensor name.
  const string& legacy_tflite_name = TfToTfLiteLegacyTensorName(tensor_name);
  if (tensor_name_map.find(legacy_tflite_name) != tensor_name_map.end()) {
    tensor_name = legacy_tflite_name;
    return OkStatus();
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
  return OkStatus();
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
  return OkStatus();
}

std::vector<int> TensorDims(const Tensor& tensor) {
  std::vector<int> dims(tensor.dims());
  for (int i = 0; i < tensor.dims(); ++i) {
    dims[i] = static_cast<int>(tensor.dim_size(i));
  }
  return dims;
}

// Create output tensors making sure they are the right size. //
Status CreateOutputTensors(
    std::unique_ptr<internal::TfLiteInterpreterWrapper>& interpreter_wrapper,
    const std::vector<string>& output_tensor_names,
    const std::map<string, int>& output_tensor_to_idx,
    std::map<int32_t, Tensor*>& tflite_idx_to_output_tensor,
    std::vector<Tensor>* output_tensors) {
  output_tensors->reserve(output_tensor_names.size());
  for (std::string tfname : output_tensor_names) {
    auto fix_status = FixTfLiteTensorName(output_tensor_to_idx, tfname);
    if (fix_status != OkStatus()) {
      return errors::Internal("Missing output TFLite tensor: ", tfname, ": ",
                              fix_status.error_message());
    }
    const int tflite_idx = output_tensor_to_idx.at(tfname);
    TensorShape tf_shape;
    const auto& interpreter = interpreter_wrapper->Get();
    const auto* tflite_tensor = interpreter->tensor(tflite_idx);
    for (int i = 0; i < tflite_tensor->dims->size; ++i) {
      tf_shape.AddDim(tflite_tensor->dims->data[i]);
    }
    DataType tf_type;
    TF_RETURN_IF_ERROR(TfLiteTypeToTfType(tflite_tensor->type, &tf_type));
    output_tensors->emplace_back(tf_type, tf_shape);
    tflite_idx_to_output_tensor[tflite_idx] = &output_tensors->back();
  }
  return OkStatus();
}

Status SetInputAndInvokeMiniBatch(
    std::unique_ptr<internal::TfLiteInterpreterWrapper>& interpreter_wrapper,
    const std::vector<int>& tflite_input_indices,
    const std::vector<std::vector<const Tensor*>>& inputs, int batch_size,
    int* fixed_batch_size) {
  auto* interpreter = interpreter_wrapper->Get();
  // Load input data from Tensorflow tensors.
  for (int i = 0; i < tflite_input_indices.size(); ++i) {
    int tflite_input_idx = tflite_input_indices[i];
    auto tflite_input_tensor = interpreter->tensor(tflite_input_idx);
    const auto& tf_input_tensors = inputs[i];
    if (tflite_input_tensor->type != kTfLiteString) {
      const Tensor* tf_input_tensor = tf_input_tensors[0];
      // concated.tensor_data() may be accessed later.
      Tensor concated;
      if (tf_input_tensors.size() > 1) {
        std::vector<Tensor> to_concatenate;
        to_concatenate.reserve(tf_input_tensors.size());
        for (const auto* t : tf_input_tensors) {
          to_concatenate.push_back(std::move(*t));
        }
        TF_RETURN_IF_ERROR(tensor::Concat(to_concatenate, &concated));
        tf_input_tensor = &concated;
      }
      auto tensor_bytes = tf_input_tensor->tensor_data();
      std::vector<int> tf_dims = TensorDims(*tf_input_tensor);
      std::vector<int> tflite_dims(
          tflite_input_tensor->dims->data,
          tflite_input_tensor->dims->data + tflite_input_tensor->dims->size);
      if (tensor_bytes.size() != tflite_input_tensor->bytes ||
          tf_dims != tflite_dims) {
        if (interpreter->ResizeInputTensor(tflite_input_idx, tf_dims) !=
            kTfLiteOk) {
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
      const bool needs_resize =
          fixed_batch_size ? batch_size > interpreter_wrapper->GetBatchSize()
                           : batch_size != interpreter_wrapper->GetBatchSize();
      if (needs_resize) {
        // std::cout << "resizing to: " << batch_size << std::endl;
        interpreter->ResizeInputTensor(tflite_input_idx, {batch_size});
        interpreter_wrapper->SetBatchSize(batch_size);
        if (interpreter->AllocateTensors() != kTfLiteOk) {
          return errors::Internal("Failed to allocate tensors");
        }
      }
      if (fixed_batch_size) {
        *fixed_batch_size = interpreter_wrapper->GetBatchSize();
      }
      TF_RETURN_IF_ERROR(interpreter_wrapper->SetStringData(
          tf_input_tensors, tflite_input_tensor, tflite_input_idx, batch_size));
    }
  }
  if (interpreter_wrapper->Invoke() != kTfLiteOk) {
    return errors::Internal("Failed to invoke TfLite interpreter");
  }
  return OkStatus();
}

Status SetMiniBatchOutput(
    std::unique_ptr<internal::TfLiteInterpreterWrapper>& interpreter_wrapper,
    const std::map<int, Tensor*>& tflite_idx_to_output_tensor,
    std::vector<Tensor>* outputs) {
  for (const auto& entry : tflite_idx_to_output_tensor) {
    Tensor* tensor = entry.second;
    const DataType tf_type = tensor->dtype();
    if (tensor->NumElements() == 0) {
      continue;
    }
    const auto* interpreter = interpreter_wrapper->Get();
    auto tflite_tensor = interpreter->tensor(entry.first);
    if (DataTypeCanUseMemcpy(tf_type)) {
      auto tensor_bytes = tensor->tensor_data();
      int offset = 0;
      size_t tflite_tensor_bytes = tflite_tensor->bytes;
      std::memcpy(const_cast<char*>(tensor_bytes.data() + offset),
                  tflite_tensor->data.raw, tflite_tensor_bytes);
    } else if (tflite_tensor->type == kTfLiteString) {
      const int string_count = tflite::GetStringCount(tflite_tensor);
      int num_strings = string_count;
      int offset = 0;
      auto str_tensors = tensor->flat<tstring>();
      for (int i = 0; i < num_strings; i++) {
        const auto& ref = tflite::GetString(tflite_tensor, i);
        str_tensors(i + offset).assign(ref.str, ref.len);
      }
    }
  }
  return OkStatus();
}

int GetModelBatchSize(const tflite::Model* model) {
  const auto* primary_subgraph = model->subgraphs()->Get(0);
  const auto* inputs = primary_subgraph->inputs();
  if (inputs->size() == 1) {
    // Only models with 1 input tensor can be batched, since SplitTFLiteTask
    // only works on a single input tensor jobs.
    const int tensor_id = inputs->Get(0);
    const auto* tensor = primary_subgraph->tensors()->Get(tensor_id);
    return tensor->shape()->Get(0);
  }
  return -1;
}

}  // namespace

// Split an input task up into multiple tasks.
Status TfLiteSession::SplitTfLiteInputTask(
    std::unique_ptr<TfLiteBatchTask>* input_task_ptr,
    int open_batch_remaining_slot, int max_batch_size,
    std::vector<std::unique_ptr<TfLiteBatchTask>>* output_tasks) {
  auto* input_task = input_task_ptr->get();
  auto split_output =
      std::make_shared<std::vector<std::unique_ptr<std::vector<Tensor>>>>();
  auto partial_status = std::make_shared<ThreadSafeStatus>();
  auto split_task_done_callback = [split_output, partial_status, input_task]() {
    // notify the input task.
    auto cleanup = gtl::MakeCleanup([done_notification = input_task->done]() {
      done_notification->Notify();
    });

    // partial status is set during actual running.
    if (!partial_status->status().ok()) {
      *input_task->status = partial_status->status();
      return;
    }

    // get the total number of tensors to concatenate (number of tasks)
    int output_size = split_output->size();
    // each split contains the same number of output tensors.
    int tensor_size = (*split_output)[0]->size();

    // for each tensor output
    for (int tensor_idx = 0; tensor_idx < tensor_size; ++tensor_idx) {
      Tensor output_tensor;  // the concatened tensor
      std::vector<Tensor> to_concatenate;
      to_concatenate.reserve(output_size);
      // for each split task concatenate the output
      for (int output_idx = 0; output_idx < output_size; ++output_idx) {
        to_concatenate.push_back(
            std::move((*(*split_output)[output_idx])[tensor_idx]));
      }
      const auto concat_status = tensor::Concat(to_concatenate, &output_tensor);
      if (!concat_status.ok()) {
        *input_task->status = concat_status;
        return;
      }
      // add the concatenated tensor to input_tasks output
      input_task->outputs->push_back(output_tensor);
    }
    *input_task->status = OkStatus();
  };

  // The Callback will be run only after all partial tasks finished.
  IncrementalBarrier barrier(std::move(split_task_done_callback));
  std::vector<int64_t> output_task_sizes;

  if (open_batch_remaining_slot > 0) {
    output_task_sizes.push_back(open_batch_remaining_slot);
    split_output->emplace_back(absl::make_unique<std::vector<Tensor>>());
  }

  for (int left_task_size = input_task->size() - open_batch_remaining_slot;
       left_task_size > 0; left_task_size -= max_batch_size) {
    int next_task_size = std::min(left_task_size, max_batch_size);
    output_task_sizes.push_back(next_task_size);
    split_output->emplace_back(absl::make_unique<std::vector<Tensor>>());
  }

  const int output_task_num = output_task_sizes.size();
  output_tasks->reserve(output_task_num);
  for (int i = 0; i < output_task_num; ++i) {
    std::unique_ptr<TfLiteBatchTask> task;
    TfLiteBatchTask::CreatePartialTfLiteBatchTask(
        input_task->input_indices, input_task->output_tensor_names,
        (*split_output)[i].get(), barrier.Inc(), partial_status.get(), &task);
    output_tasks->push_back(std::move(task));
  }

  for (int i = 0; i < input_task->inputs.size(); ++i) {
    const Tensor& input = input_task->inputs[i];
    std::vector<Tensor> split_tensors;
    auto status = tensor::Split(input, output_task_sizes, &split_tensors);
    if (status != OkStatus()) {
      return status;
    }
    for (int output_idx = 0; output_idx < output_task_num; ++output_idx) {
      auto& output_task = (*output_tasks)[output_idx];
      output_task->inputs.push_back(std::move(split_tensors[output_idx]));
    }
  }
  return OkStatus();
}

Status TfLiteSession::CreateDefaultBasicBatchScheduler(
    const BasicBatchScheduler<TfLiteBatchTask>::Options& options,
    std::function<void(std::unique_ptr<Batch<TfLiteBatchTask>>)>
        process_batch_callback,
    std::unique_ptr<BasicBatchScheduler<TfLiteBatchTask>>* batch_scheduler) {
  std::unique_ptr<BasicBatchScheduler<TfLiteBatchTask>> basic_batch_scheduler;
  TF_RETURN_IF_ERROR(BasicBatchScheduler<TfLiteBatchTask>::Create(
      options, process_batch_callback, &basic_batch_scheduler));
  *batch_scheduler = std::move(basic_batch_scheduler);
  return OkStatus();
}

Status TfLiteSession::SetScheduler(
    const SchedulerCreator& scheduler_creator,
    const BasicBatchScheduler<TfLiteBatchTask>::Options& options) {
  use_fixed_batch_size_ = true;
  scheduler_options_ = options;
  auto bound_scheduler_creator = absl::bind_front(
      &TfLiteSession::CreateDefaultBasicBatchScheduler, scheduler_options_);
  return bound_scheduler_creator(
      [this](std::unique_ptr<Batch<TfLiteBatchTask>> batch) {
        this->ProcessBatch(std::move(batch));
      },
      &scheduler_);
}

Status TfLiteSession::Create(string&& buffer, const SessionOptions& options,
                             int num_pools, int num_interpreters_per_pool,
                             std::unique_ptr<TfLiteSession>* tflite_session,
                             ::google::protobuf::Map<string, SignatureDef>* signatures) {
  auto model = tflite::FlatBufferModel::BuildFromModel(
      flatbuffers::GetRoot<tflite::Model>(buffer.data()));
  if (model == nullptr) {
    return errors::InvalidArgument("Cannot build FlatBufferModel from buffer.");
  }

  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::ops::custom::AddParseExampleOp(&resolver);

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
  if (status != OkStatus()) {
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

  const int num_interpreters = std::max(1, num_pools);
  const int model_batch_size = GetModelBatchSize(model->GetModel());

  std::unique_ptr<internal::TfLiteInterpreterPool> interpreter_pool;
  TF_RETURN_IF_ERROR(
      internal::TfLiteInterpreterPool::CreateTfLiteInterpreterPool(
          model.get(), options, num_interpreters, interpreter_pool));

  tflite_session->reset(new TfLiteSession(
      std::move(input_tensor_to_index), std::move(output_tensor_to_index),
      std::move(buffer), std::move(model), std::move(interpreter_pool)));

  if (num_interpreters_per_pool > 1) {
    const int default_allowed_batch =
        (internal::kInitialBatchSize + num_interpreters_per_pool - 1) /
        num_interpreters_per_pool;
    const int min_allowed_batch =
        model_batch_size > 1 ? model_batch_size : default_allowed_batch;
    const int max_enqueued_batches = num_interpreters * 100;
    BasicBatchScheduler<TfLiteBatchTask>::Options scheduler_options;
    scheduler_options.num_batch_threads = num_interpreters;
    scheduler_options.max_batch_size = internal::kInitialBatchSize;
    scheduler_options.enable_large_batch_splitting = true;
    scheduler_options.max_execution_batch_size = min_allowed_batch;
    scheduler_options.max_enqueued_batches = max_enqueued_batches;
    scheduler_options.split_input_task_func = SplitTfLiteInputTask;
    TF_RETURN_IF_ERROR(
        (*tflite_session)
            ->SetScheduler(&TfLiteSession::CreateDefaultBasicBatchScheduler,
                           scheduler_options));
  }
  return OkStatus();
}

TfLiteSession::TfLiteSession(
    std::map<string, int>&& input_tensor_to_index,
    std::map<string, int>&& output_tensor_to_index, string&& buffer,
    std::unique_ptr<tflite::FlatBufferModel> model,
    std::unique_ptr<internal::TfLiteInterpreterPool> interpreter_pool)
    : input_tensor_to_index_(std::move(input_tensor_to_index)),
      output_tensor_to_index_(std::move(output_tensor_to_index)),
      model_serialized_bytes_(std::move(buffer)),
      model_(std::move(model)),
      interpreter_pool_(std::move(interpreter_pool)) {}

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

Status TfLiteSession::RunInternal(
    const std::vector<int>& tflite_input_indices,
    const std::vector<std::vector<const Tensor*>>& merged_inputs,
    const std::vector<string>& output_tensor_names,
    std::vector<Tensor>* combined_outputs, int batch_size,
    int* fixed_batch_size) {
#define RETURN_POOL_IF_ERROR(...)                                   \
  do {                                                              \
    ::tensorflow::Status _status = (__VA_ARGS__);                   \
    if (TF_PREDICT_FALSE(!_status.ok())) {                          \
      interpreter_pool_->ReturnInterpreter(std::move(interpreter)); \
      return _status;                                               \
    }                                                               \
  } while (0);
  auto interpreter = interpreter_pool_->GetInterpreter();
  RETURN_POOL_IF_ERROR(
      SetInputAndInvokeMiniBatch(interpreter, tflite_input_indices,
                                 merged_inputs, batch_size, fixed_batch_size));

  // Create return tensors and map the tflite tensor index to the
  // index of the created tensor.
  std::map<int32_t, Tensor*> tflite_idx_to_output_tensor;
  RETURN_POOL_IF_ERROR(CreateOutputTensors(
      interpreter, output_tensor_names, output_tensor_to_index_,
      tflite_idx_to_output_tensor, combined_outputs));

  // Set the contents of the return tensors.
  RETURN_POOL_IF_ERROR(SetMiniBatchOutput(
      interpreter, tflite_idx_to_output_tensor, combined_outputs));

#undef RETURN_POOL_IF_ERROR
  interpreter_pool_->ReturnInterpreter(std::move(interpreter));
  return OkStatus();
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
  outputs->reserve(output_tensor_names.size());
  if (!scheduler_) {
    std::vector<int> input_indices;
    std::vector<std::vector<const Tensor*>> inputs;
    for (const auto entry : tflite_idx_to_input_tensor) {
      const auto& tf_tensor = *entry.second;
      inputs.push_back({&tf_tensor});
      input_indices.push_back(entry.first);
    }
    const int batch_size =
        inputs.empty() || inputs[0].empty() ? 1 : inputs[0][0]->dim_size(0);
    return RunInternal(input_indices, inputs, output_tensor_names, outputs,
                       batch_size);
  }
  Notification done;
  Status status;
  std::unique_ptr<TfLiteBatchTask> task;
  TfLiteBatchTask::CreateTfLiteBatchTask(&output_tensor_names, outputs, &done,
                                         &status, &task);
  for (const auto entry : tflite_idx_to_input_tensor) {
    task->input_indices.push_back(entry.first);
    task->inputs.push_back(std::move(*entry.second));
  }
  TF_RETURN_IF_ERROR(scheduler_->Schedule(&task));
  done.WaitForNotification();
  return status;
}

Status TfLiteSession::ListDevices(std::vector<DeviceAttributes>* response) {
  return errors::Unimplemented("ListDevices is not yet supported.");
}

Status MergeInputTensors(const Batch<TfLiteBatchTask>& batch,
                         std::vector<std::vector<const Tensor*>>* merged_inputs,
                         int* batch_size) {
  if (batch.num_tasks() < 1) {
    return errors::Internal("Batch size expected to be positive; was ",
                            batch.num_tasks());
  }
  const int tensors_per_task = batch.task(0).inputs.size();
  *batch_size = 0;
  // each entry in merged_inputs is a list of task tensors.
  for (int i = 0; i < tensors_per_task; ++i) {
    merged_inputs->emplace_back();
    std::vector<const Tensor*>& tensors_to_merge = merged_inputs->back();
    for (int j = 0; j < batch.num_tasks(); ++j) {
      const std::vector<Tensor>& inputs = batch.task(j).inputs;
      tensors_to_merge.push_back(&(inputs[i]));
      if (i == 0) {
        if (inputs[i].dims()) {
          *batch_size += inputs[i].dim_size(0);
        }
      }
    }
  }
  return OkStatus();
}

Status SplitOutputTensors(const std::vector<Tensor>& combined_outputs,
                          Batch<TfLiteBatchTask>* batch, int batch_size) {
  std::vector<int64_t> task_sizes(batch->num_tasks());
  int total_size = 0;
  for (int i = 0; i < batch->num_tasks(); ++i) {
    const int task_size = batch->task(i).size();
    task_sizes[i] = task_size;
    total_size += task_size;
  }

  if (total_size < batch_size) {
    task_sizes.push_back(batch_size - total_size);
  }

  for (int i = 0; i < combined_outputs.size(); i++) {
    const auto& output_tensor = combined_outputs[i];
    std::vector<Tensor> split_tensor;
    const Status split_status =
        tensor::Split(output_tensor, task_sizes, &split_tensor);
    if (!split_status.ok()) {
      return errors::Internal("Tensor split operation failed: ",
                              split_status.ToString());
    }
    for (int j = 0; j < batch->num_tasks(); ++j) {
      TfLiteBatchTask& task = *(batch->mutable_task(j));
      task.set_output(split_tensor[j]);
    }
  }

  return OkStatus();
}

void TfLiteSession::ProcessBatch(
    std::unique_ptr<Batch<TfLiteBatchTask>> batch) {
  // As a possible performance optimization, consider overlapping the tensor
  // concatenation with waiting for the batch to close (i.e. do the
  // concatenation incrementally as tasks stream into the batch).
  batch->WaitUntilClosed();

  if (batch->empty()) {
    return;
  }

  const uint64_t dequeue_time_micros = EnvTime::NowMicros();

  // Regardless of the outcome, we need to propagate the status to the
  // individual tasks and signal that they are done. We use MakeCleanup() to
  // ensure that this happens no matter how we exit the method below.
  Status status;
  auto finally = gtl::MakeCleanup([&status, &batch] {
    for (int i = 0; i < batch->num_tasks(); ++i) {
      TfLiteBatchTask* task = batch->mutable_task(i);
      if (task->is_partial) {
        task->partial_status->Update(status);
        task->done_callback();
      } else {
        *batch->mutable_task(i)->status = status;
        batch->mutable_task(i)->done->Notify();
      }
    }
  });

  // Make sure we have at least one task that hasn't exceeded its timeout from
  // queue time alone, and find the latest task deadline which we'll use for the
  // overall batch.
  bool all_tasks_timeout_exceeded = true;
  uint64_t batch_deadline_micros = 0;
  for (int i = 0; i < batch->num_tasks(); ++i) {
    const TfLiteBatchTask& task = batch->task(i);
    // If the caller doesn't populate RunOptions, the timeout is 0 by default.
    // Interpret that as "no timeout".
    if (task.run_options.timeout_in_ms() <= 0) {
      all_tasks_timeout_exceeded = false;
      break;
    }
    const int64_t task_timeout_micros = task.run_options.timeout_in_ms() * 1000;
    const uint64_t task_deadline_micros =
        task.enqueue_time_micros + task_timeout_micros;
    if (task_deadline_micros > dequeue_time_micros) {
      all_tasks_timeout_exceeded = false;
      if (task_deadline_micros > batch_deadline_micros) {
        batch_deadline_micros = task_deadline_micros;
      }
    }
  }
  if (all_tasks_timeout_exceeded) {
    status = Status(error::RESOURCE_EXHAUSTED,
                    "Run() timeout exceeded while waiting in batching queue");
    return;
  }

  std::vector<std::vector<const Tensor*>> merged_inputs;
  int batch_size = 0;
  status = MergeInputTensors(*batch, &merged_inputs, &batch_size);
  if (!status.ok()) {
    return;
  }
  std::vector<Tensor> combined_outputs;
  const auto& tflite_input_indices = batch->task(0).input_indices;
  auto& output_tensor_names = batch->task(0).output_tensor_names;
  int fixed_batch_size = batch_size;
  status = RunInternal(tflite_input_indices, merged_inputs,
                       *output_tensor_names, &combined_outputs, batch_size,
                       use_fixed_batch_size_ ? &fixed_batch_size : nullptr);
  if (!status.ok()) {
    return;
  }
  // The size of the batch might be smaller than the fixed_batch_size.
  status = SplitOutputTensors(combined_outputs, batch.get(), fixed_batch_size);
}

}  // namespace serving
}  // namespace tensorflow
