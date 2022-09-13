/* Copyright 2020 Google Inc. All Rights Reserved.

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

#include "tensorflow_serving/servables/tensorflow/tflite_interpreter_pool.h"

#include <limits>
#include <memory>
#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/lite/external_cpu_backend_context.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/parse_example/parse_example.h"
#include "tensorflow/lite/kernels/register.h"

namespace tensorflow {
namespace serving {
namespace internal {

TfLiteInterpreterWrapper::TfLiteInterpreterWrapper(
    std::unique_ptr<tflite::ExternalCpuBackendContext> external_context,
    std::unique_ptr<tflite::Interpreter> interpreter)
    :
#ifdef TFLITE_PROFILE
      external_context_(std::move(external_context)),
      interpreter_(std::move(interpreter)),
      max_num_entries_(TFLITE_PROFILE_EVENTS),
      profiler_(max_num_entries_)
#else
      external_context_(std::move(external_context)),
      interpreter_(std::move(interpreter))
#endif
{
#ifdef TFLITE_PROFILE
  interpreter_->SetProfiler(&profiler_);
#endif
  for (const int& idx : interpreter_->inputs()) {
    const auto* tflite_tensor = interpreter_->tensor(idx);
    if (tflite_tensor->type == kTfLiteString) {
      tensor_buffer_.emplace(idx,
                             std::unique_ptr<char>(tflite_tensor->data.raw));
      tensor_buffer_max_bytes_[idx] = 0;
    }
  }
}

tensorflow::Status TfLiteInterpreterWrapper::SetStringData(
    const std::vector<const Tensor*>& tensors, TfLiteTensor* tflite_tensor,
    int tensor_index, int batch_size) {
  // Format of the buffer for tflite:
  //   [0] number of strings (int32_t)
  //   [4] offset of each string (int32_t)
  //   [sizeof(int32_t) * (num_strings + 1)]] total size of strings
  //   [sizeof(int32_t) * (num_strings + 2)] batch.data()
  int32_t num_strings = batch_size;
  offset_.clear();
  size_t total_size = 0;
  offset_.push_back(static_cast<int32_t>(total_size));
  for (const auto& tensor : tensors) {
    const auto& flat = tensor->flat<tstring>();
    for (int i = 0; i < flat.size(); ++i) {
      total_size += flat(i).size();
      offset_.push_back(static_cast<int32_t>(total_size));
    }
  }
  size_t required_bytes = total_size + sizeof(int32_t) * (num_strings + 2);
  if (tensor_buffer_.find(tensor_index) == tensor_buffer_.end()) {
    return errors::Internal("Tensor input for index not found: ", tensor_index);
  }
  if (required_bytes > tensor_buffer_max_bytes_[tensor_index]) {
    if (tflite_tensor->data.raw) {
      free(tflite_tensor->data.raw);
    }
    tflite_tensor->data.raw = reinterpret_cast<char*>(malloc(required_bytes));
    tensor_buffer_max_bytes_[tensor_index] = required_bytes;
  }
  tensor_buffer_[tensor_index].reset(tflite_tensor->data.raw);
  memcpy(tensor_buffer_[tensor_index].get(), &num_strings, sizeof(int32_t));
  int32_t start = sizeof(int32_t) * (num_strings + 2);
  for (size_t i = 0; i < offset_.size(); i++) {
    size_t size_offset_i = start + offset_[i];
    if (size_offset_i > std::numeric_limits<int32_t>::max()) {
      return errors::Internal("Invalid size, string input too large:",
                              size_offset_i);
    }
    int32_t offset_i = static_cast<int32_t>(size_offset_i);
    memcpy(tensor_buffer_[tensor_index].get() + sizeof(int32_t) * (i + 1),
           &offset_i, sizeof(int32_t));
  }
  for (const auto& tensor : tensors) {
    const auto& flat = tensor->flat<tstring>();
    for (int i = 0; i < flat.size(); ++i) {
      memcpy(tensor_buffer_[tensor_index].get() + start, flat(i).data(),
             flat(i).size());
      start += flat(i).size();
    }
  }

  // tflite_tensor will take ownership of the pointer.
  tflite_tensor->data.raw = tensor_buffer_[tensor_index].release();
  tflite_tensor->bytes = required_bytes;
  tflite_tensor->allocation_type = kTfLiteDynamic;
  return OkStatus();
}

TfLiteStatus TfLiteInterpreterWrapper::Invoke() {
#ifdef TFLITE_PROFILE
  if (invocation_count_ > 0) {
    profiler_.Reset();
    profiler_.StartProfiling();
  }
#endif
  auto status = interpreter_->Invoke();
#ifdef TFLITE_PROFILE
  if (invocation_count_ > 0) {
    profiler_.StopProfiling();
    auto profile_events = profiler_.GetProfileEvents();
    run_summarizer_.ProcessProfiles(profile_events, *interpreter_);
  }
  if (invocation_count_++ >= MAX_PROFILE_EVENTS) {
    WriteProfileData();
    run_summarizer_.Clear();
    invocation_count_ = 0;
  }
#endif
  return status;
}

tensorflow::Status TfLiteInterpreterWrapper::CreateTfLiteInterpreterWrapper(
    const tflite::FlatBufferModel& model,
    const tensorflow::SessionOptions& options,
    std::unique_ptr<TfLiteInterpreterWrapper>& wrapper) {
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::ops::custom::AddParseExampleOp(&resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;

  // Use an initial batch_size of 1, will be resized later.
  const int batch_size = 1;
  // Use a single thread to reduce contention across sessions.
  const int num_threads = 1;

  if (tflite::InterpreterBuilder(model, resolver)(&interpreter, num_threads) !=
      kTfLiteOk) {
    return errors::Internal(
        "Failed to create a TFLite interpreter with the given model");
  }
  std::unique_ptr<tflite::ExternalCpuBackendContext> external_context(
      new tflite::ExternalCpuBackendContext());
  std::unique_ptr<tflite::CpuBackendContext> cpu_backend_context(
      new tflite::CpuBackendContext());
  cpu_backend_context->SetUseCaching(true);
  cpu_backend_context->SetMaxNumThreads(num_threads);
  external_context->set_internal_backend_context(
      std::move(cpu_backend_context));
  interpreter->SetExternalContext(kTfLiteCpuBackendContext,
                                  external_context.get());
  const int idx = interpreter->inputs()[0];
  const auto* tensor = interpreter->tensor(idx);
  if (tensor->type == kTfLiteString) {
    if (interpreter->ResizeInputTensor(idx, {batch_size}) != kTfLiteOk) {
      return errors::Internal("Failed to resize input");
    }
  }
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    return errors::Internal("Failed to allocate tensors");
  }
  wrapper.reset(new TfLiteInterpreterWrapper(std::move(external_context),
                                             std::move(interpreter)));
  return tensorflow::OkStatus();
}

tensorflow::Status TfLiteInterpreterPool::CreateTfLiteInterpreterPool(
    const tflite::FlatBufferModel* model,
    const tensorflow::SessionOptions& options, int pool_size,
    std::unique_ptr<TfLiteInterpreterPool>& interpreter_pool) {
  std::vector<std::unique_ptr<TfLiteInterpreterWrapper>> interpreters(
      pool_size);
  for (int i = 0; i < pool_size; i++) {
    auto& wrapper = interpreters[i];
    TF_RETURN_IF_ERROR(TfLiteInterpreterWrapper::CreateTfLiteInterpreterWrapper(
        *model, options, wrapper));
  }
  interpreter_pool.reset(new TfLiteInterpreterPool(std::move(interpreters)));
  return tensorflow::OkStatus();
}

}  // namespace internal
}  // namespace serving
}  // namespace tensorflow
