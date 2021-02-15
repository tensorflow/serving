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

#include "tensorflow/core/platform/errors.h"
#include "tensorflow/lite/kernels/hashtable/hashtable_ops.h"
#include "tensorflow/lite/kernels/parse_example/parse_example.h"
#include "tensorflow/lite/kernels/register.h"

namespace tensorflow {
namespace serving {
namespace internal {

TfLiteInterpreterWrapper::TfLiteInterpreterWrapper(
    std::unique_ptr<tflite::Interpreter> interpreter)
    :
#ifdef TFLITE_PROFILE
      interpreter_(std::move(interpreter)),
      fixed_batch_size_(fixed_batch_size),
      max_num_entries_(TFLITE_PROFILE_EVENTS),
      profiler_(max_num_entries_)
#else
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
    const gtl::ArraySlice<tensorflow::tstring>& batch,
    TfLiteTensor* tflite_tensor, int tensor_index) {
  // Format of the buffer for tflite:
  //   [0] number of strings (int32_t)
  //   [4] offset of each string (int32_t)
  //   [sizeof(int32_t) * (num_strings + 1)]] total size of strings
  //   [sizeof(int32_t) * (num_strings + 2)] batch.data()
  int32_t num_strings = batch.size();
  offset_.clear();
  size_t total_size = 0;
  offset_.push_back(static_cast<int32_t>(total_size));
  for (int i = 0; i < num_strings; i++) {
    total_size += batch[i].size();
    offset_.push_back(static_cast<int32_t>(total_size));
  }
  size_t required_bytes = total_size + sizeof(int32_t) * (num_strings + 2);
  if (tensor_buffer_.find(tensor_index) == tensor_buffer_.end()) {
    return errors::Internal("Tensor input for index not found: ", tensor_index);
  }
  if (tensor_buffer_max_bytes_[tensor_index] > 0 &&
      required_bytes > tensor_buffer_max_bytes_[tensor_index]) {
    tensor_buffer_max_bytes_[tensor_index] = 0;
  }

  if (tensor_buffer_max_bytes_[tensor_index] == 0) {
    tensor_buffer_[tensor_index].reset(
        reinterpret_cast<char*>(malloc(required_bytes)));
    tensor_buffer_max_bytes_[tensor_index] = required_bytes;
  } else {
    tensor_buffer_[tensor_index].reset(tflite_tensor->data.raw);
  }
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
  for (int i = 0; i < num_strings; i++) {
    memcpy(tensor_buffer_[tensor_index].get() + start, batch[i].data(),
           batch[i].size());
    start += batch[i].size();
  }

  // tflite_tensor will take ownership of the pointer.
  tflite_tensor->data.raw = tensor_buffer_[tensor_index].release();
  tflite_tensor->bytes = required_bytes;
  tflite_tensor->allocation_type = kTfLiteDynamic;
  return Status::OK();
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

tensorflow::Status TfLiteInterpreterPool::CreateTfLiteInterpreterPool(
    const tflite::FlatBufferModel& model, bool run_in_caller,
    bool use_batch_parallelism, int batch_pool_size, int id,
    const tensorflow::SessionOptions& options,
    std::unique_ptr<TfLiteInterpreterPool>& pool) {
  // If can't use_batch_parallelism or pool size is 1,
  // just use 1 interpreter and run in caller.
  if (!use_batch_parallelism || batch_pool_size == 1) {
    run_in_caller = true;
    batch_pool_size = 1;
    use_batch_parallelism = false;
  }
  std::unique_ptr<tensorflow::thread::ThreadPool> thread_pool;
  int num_interpreters = run_in_caller ? 1 : 0;
  if (!run_in_caller || use_batch_parallelism) {
    thread_pool.reset(new tensorflow::thread::ThreadPool(
        options.env, tensorflow::ThreadOptions(),
        absl::StrCat(kTfLiteThreadPoolName, id), batch_pool_size, false,
        nullptr));
    num_interpreters += thread_pool->NumThreads();
  }

  // TODO(b/140959776): Add support for non-builtin ops (flex or custom ops).
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::ops::custom::AddParseExampleOp(&resolver);
  // TODO(b/165643512): Remove adding Hashtable to resolver by default.
  tflite::ops::custom::AddHashtableOps(&resolver);
  int fixed_batch_size = 1;
  if (use_batch_parallelism) {
    if (num_interpreters < 1) {
      return errors::InvalidArgument(
          "CreateTfLiteInterpreterPool requested ",
          "invalid number of interpreters: ", num_interpreters);
    }
    fixed_batch_size =
        (kInitialBatchSize + num_interpreters - 1) / num_interpreters;
  }
  std::vector<std::unique_ptr<TfLiteInterpreterWrapper>> interpreters;

  for (int i = 0; i < num_interpreters; i++) {
    std::unique_ptr<tflite::Interpreter> interpreter;
    if (tflite::InterpreterBuilder(model, resolver)(
            &interpreter, /*num_threads=*/1) != kTfLiteOk) {
      return errors::Internal(
          "Failed to create a TFLite interpreter with the given model");
    }
    const int idx = interpreter->inputs()[0];
    const auto* tensor = interpreter->tensor(idx);
    if (tensor->type == kTfLiteString) {
      interpreter->ResizeInputTensor(idx, {fixed_batch_size});
    }
    if (interpreter->AllocateTensors() != kTfLiteOk) {
      return errors::Internal("Failed to allocate tensors");
    }
    interpreters.push_back(
        std::make_unique<TfLiteInterpreterWrapper>(std::move(interpreter)));
    interpreters.back()->SetMiniBatchSize(fixed_batch_size);
  }
  pool.reset(new TfLiteInterpreterPool(
      id, std::move(interpreters), std::move(thread_pool), num_interpreters,
      fixed_batch_size, use_batch_parallelism));
  return tensorflow::Status::OK();
}

TfLiteInterpreterPool::TfLiteInterpreterPool(
    int id, std::vector<std::unique_ptr<TfLiteInterpreterWrapper>> interpreters,
    std::unique_ptr<tensorflow::thread::ThreadPool> thread_pool,
    int num_interpreters, int fixed_batch_size, bool use_batch_parallelism)
    : id_(id),
      interpreters_(std::move(interpreters)),
      thread_pool_(std::move(thread_pool)),
      num_interpreters_(num_interpreters),
      fixed_batch_size_(fixed_batch_size),
      use_batch_parallelism_(use_batch_parallelism) {}

std::unique_ptr<TfLiteInterpreterWrapper>&
TfLiteInterpreterPool::GetInterpreter(int interpreter_idx) {
  return interpreters_[interpreter_idx];
}

namespace {

bool IsBatchParallelizable(const tflite::Model* model) {
  auto tflite_model = absl::make_unique<tflite::ModelT>();
  model->UnPackTo(tflite_model.get(), nullptr);
  if (tflite_model->subgraphs.empty()) {
    return false;
  }
  const auto& subgraph = tflite_model->subgraphs[0];
  if (subgraph->inputs.size() != 1) {
    return false;
  }
  int input_tensor_id = subgraph->inputs[0];
  const std::vector<string> supported_ops = {"ParseExample", "ParseExampleV2"};
  for (size_t op_idx = 0; op_idx < subgraph->operators.size(); op_idx++) {
    tflite::OperatorT* op = subgraph->operators[op_idx].get();
    if (std::find(op->inputs.begin(), op->inputs.end(), input_tensor_id) !=
        op->inputs.end()) {
      const std::string& custom_code =
          tflite_model->operator_codes[op->opcode_index]->custom_code;
      return std::find(supported_ops.begin(), supported_ops.end(),
                       custom_code) != supported_ops.end();
    }
  }
  return false;
}

}  // namespace

tensorflow::Status TfLiteSessionPool::CreateTfLiteSessionPool(
    const tflite::FlatBufferModel* model,
    const tensorflow::SessionOptions& options, bool run_in_caller,
    int pool_size, int batch_pool_size,
    std::unique_ptr<TfLiteSessionPool>& tflite_session_pool) {
  bool use_batch_parallelism = IsBatchParallelizable(model->GetModel());
  std::vector<std::unique_ptr<TfLiteInterpreterPool>> pools;
  for (int i = 0; i < pool_size; i++) {
    std::unique_ptr<TfLiteInterpreterPool> pool;
    TF_RETURN_IF_ERROR(TfLiteInterpreterPool::CreateTfLiteInterpreterPool(
        *model, run_in_caller, use_batch_parallelism, batch_pool_size, i,
        options, pool));
    pools.push_back(std::move(pool));
  }
  tflite_session_pool.reset(new TfLiteSessionPool(std::move(pools)));
  return tensorflow::Status::OK();
}

}  // namespace internal
}  // namespace serving
}  // namespace tensorflow
