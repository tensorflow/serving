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

#ifndef TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_TFLITE_INTERPRETER_POOL_H_
#define TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_TFLITE_INTERPRETER_POOL_H_

#include <map>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/model.h"
#ifdef TFLITE_PROFILE
#ifndef TFLITE_PROFILE_EVENTS
#define TFLITE_PROFILE_EVENTS 2000
#endif
#include "tensorflow/lite/profiling/buffered_profiler.h"
#include "tensorflow/lite/profiling/profile_summarizer.h"
#include "tensorflow/lite/profiling/profile_summary_formatter.h"
#endif
#include "tensorflow/lite/string_util.h"

namespace tensorflow {
namespace serving {
namespace internal {

constexpr int kInitialBatchSize = 500;

class TfLiteInterpreterWrapper {
  // Wrapper class for a single TfLite Interpreter for use in an interpreter
  // pool.
 public:
  // Create an interpreter and external context and wrap it in the class
  // for use with an InterpreterPool.
  static Status CreateTfLiteInterpreterWrapper(
      const tflite::FlatBufferModel& model,
      const tensorflow::SessionOptions& options,
      std::unique_ptr<TfLiteInterpreterWrapper>& wrapper);

  // Constructor for wrapper takes only an initialized interpreter.
  TfLiteInterpreterWrapper(
      std::unique_ptr<tflite::ExternalCpuBackendContext> external_context,
      std::unique_ptr<tflite::Interpreter> interpreter);

  TfLiteInterpreterWrapper(std::unique_ptr<tflite::Interpreter> interpreter)
      : TfLiteInterpreterWrapper(nullptr, std::move(interpreter)) {}

  // Returns the underlying interpreter.
  tflite::Interpreter* Get() { return interpreter_.get(); }

  // Get the allocated batch size of the interpreter.
  int GetBatchSize() { return batch_size_; }

  // Set the batch size.
  void SetBatchSize(int batch_size) { batch_size_ = batch_size; }

  // Invokes the interpreter.
  TfLiteStatus Invoke();
#ifdef TFLITE_PROFILE
  void WriteOutput(const std::string& header, const string& data,
                   std::ostream* stream) {
    (*stream) << header << std::endl;
    (*stream) << data << std::endl;
  }

  void WriteProfileData() {
    if (run_summarizer_.HasProfiles()) {
      WriteOutput("Operator-wise Profiling Info for Regular Benchmark Runs:",
                  run_summarizer_.GetOutputString(), &std::cout);
    }
  }
#endif

  // Sets the contents of the internal buffer _tensor_buffer_ to the tflite
  // formatted string buffer equivalent stored in `batch` and sets
  // raw pointer of `tflite_tensor` to the internal buffer. If the required
  // size is larger than the current size, will allocate new memory and
  // free the existing buffer.
  tensorflow::Status SetStringData(const std::vector<const Tensor*>& tensors,
                                   TfLiteTensor* tflite_tensor,
                                   int tensor_index, int batch_size);

 private:
  // External cpu context to enable caching.
  std::unique_ptr<tflite::ExternalCpuBackendContext> external_context_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
  int batch_size_ = 1;
  std::map<int, std::unique_ptr<char>> tensor_buffer_;
  std::map<int, size_t> tensor_buffer_max_bytes_;
  std::vector<int32_t> offset_;
#ifdef TFLITE_PROFILE
  int max_num_entries_;
  tflite::profiling::ProfileSummarizer run_summarizer_;
  tflite::profiling::BufferedProfiler profiler_;
  int invocation_count_ = 0;
#endif
};

// Contains a vector of TfLiteInterpreterWrapper, which are protected by mutex.
// When GetInterpreter is called, will either release a unique ptr to the
// caller or block if the vector is empty.
class TfLiteInterpreterPool {
 public:
  // Creates a TfLiteSessionPool with model, session options,
  // pool_size number of interpreters.
  static tensorflow::Status CreateTfLiteInterpreterPool(
      const tflite::FlatBufferModel* model,
      const tensorflow::SessionOptions& options, int pool_size,
      std::unique_ptr<TfLiteInterpreterPool>& interpreter_pool);

  // Returns a TFLite interpreter wrapper object. Caller may *block* waiting for
  // a free interpreter pool to be available.
  std::unique_ptr<TfLiteInterpreterWrapper> GetInterpreter() {
    auto interpreter_available = [this]() ABSL_SHARED_LOCKS_REQUIRED(mutex_) {
      return !this->available_.empty();
    };
    mutex_.LockWhen(absl::Condition(&interpreter_available));
    auto pool = std::move(available_.back());
    available_.pop_back();
    mutex_.Unlock();
    return pool;
  }

  // Returns an interpreter wrapper to the available pool.
  void ReturnInterpreter(
      std::unique_ptr<TfLiteInterpreterWrapper> interpreter) {
    absl::MutexLock l(&mutex_);
    available_.emplace_back(std::move(interpreter));
  }

 private:
  TfLiteInterpreterPool(
      std::vector<std::unique_ptr<TfLiteInterpreterWrapper>> interpreters)
      : available_(std::move(interpreters)) {}
  mutable absl::Mutex mutex_;
  std::vector<std::unique_ptr<TfLiteInterpreterWrapper>> available_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace internal
}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_TFLITE_INTERPRETER_POOL_H_
