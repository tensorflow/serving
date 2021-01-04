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

#include <memory>
#include <vector>

#include "tensorflow/lite/kernels/hashtable/hashtable_ops.h"
#include "tensorflow/lite/kernels/register.h"

namespace tensorflow {
namespace serving {
namespace internal {

Status TfLiteInterpreterPool::CreateTfLiteInterpreterPool(
    const tflite::FlatBufferModel& model, int num_interpreters,
    std::unique_ptr<TfLiteInterpreterPool>& pool) {
  // TODO(b/140959776): Add support for non-builtin ops (flex or custom ops).
  tflite::ops::builtin::BuiltinOpResolver resolver;
  // TODO(b/165643512): Remove adding Hashtable to resolver by default.
  tflite::ops::custom::AddHashtableOps(&resolver);
  num_interpreters = std::max(num_interpreters, 1);

  std::vector<std::unique_ptr<tflite::Interpreter>> interpreters;
  interpreters.reserve(num_interpreters);
  for (int i = 0; i < num_interpreters; i++) {
    std::unique_ptr<tflite::Interpreter> interpreter;
    if (tflite::InterpreterBuilder(model, resolver)(&interpreter) !=
        kTfLiteOk) {
      return errors::Internal(
          "Failed to create a TFLite interpreter with the given model");
    }
    if (interpreter->AllocateTensors() != kTfLiteOk) {
      return errors::Internal("Failed to allocate tensors");
    }
    interpreters.push_back(std::move(interpreter));
  }

  pool.reset(new TfLiteInterpreterPool(std::move(interpreters)));
  return Status::OK();
}

TfLiteInterpreterPool::TfLiteInterpreterPool(
    std::vector<std::unique_ptr<tflite::Interpreter>> interpreters)
    : interpreters_(std::move(interpreters)) {
  auto num_interpreters = interpreters_.size();
  available_.reserve(num_interpreters);
  for (int i = 0; i < num_interpreters; i++) {
    available_.push_back(interpreters_[i].get());
  }
}

std::unique_ptr<TfLiteInterpreterWrapper>
TfLiteInterpreterPool::GetInterpreter() {
  auto interpreter_available = [this]() ABSL_SHARED_LOCKS_REQUIRED(mutex_) {
    return !this->available_.empty();
  };
  mutex_.LockWhen(absl::Condition(&interpreter_available));

  tflite::Interpreter* interpreter = available_.back();
  available_.pop_back();
  mutex_.Unlock();
  auto interpreter_wrapper =
      std::make_unique<TfLiteInterpreterWrapper>(interpreter, this);
  return interpreter_wrapper;
}

void TfLiteInterpreterPool::ReturnInterpreter(
    tflite::Interpreter* interpreter) {
  absl::MutexLock l(&mutex_);
  available_.push_back(interpreter);
}

}  // namespace internal
}  // namespace serving
}  // namespace tensorflow
