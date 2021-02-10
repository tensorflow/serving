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
#include "tensorflow/core/common_runtime/process_util.h"
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

constexpr char kTfLiteThreadPoolName[] = "tflite_compute";

constexpr int kInitialBatchSize = 500;

class TfLiteInterpreterWrapper;

// A TFLite batch pool to maintain pre-created tflite::Interpreter
// objects.
class TfLiteInterpreterPool {
 public:
  // Create a vector of TfLiteInterpreterWrapper objects with the given TFLite
  // FlatBufferModel `model`.
  // The number of created interpreters will depend on batch_pool_size,
  // whether the calling thread requires a  interpreter, `run_in_caller`, and
  // whether the model can make use of batch parallelism.
  // If the model cannot make use of batch parallelism, only
  // a single interpreter will be created. Otherwise, the setting is set by
  // batch_pool_size + 1 if run_in_caller is true.
  // The model must outlive the returned pool instance.
  static Status CreateTfLiteInterpreterPool(
      const tflite::FlatBufferModel& model, bool run_in_caller,
      bool use_batch_parallelism, int batch_pool_size, int id,
      const tensorflow::SessionOptions& options,
      std::unique_ptr<TfLiteInterpreterPool>& pool);

  // The allocated batch size of the TfLite interpreters.
  int FixedBatchSize() { return fixed_batch_size_; }

  // Returns an interpreter wrapper from the pool at the given index.
  std::unique_ptr<TfLiteInterpreterWrapper>& GetInterpreter(
      int interpreter_idx);

  // The Id of the interpreterwrapper object, used to target a specific
  // interpreterwrapper.
  const int Id() { return id_; }

  // Number of interpreters.
  const int NumInterpreters() { return num_interpreters_; }

  // Returns a ThreadPool for use with each interpreter, will be null
  // if using run_in_caller and only 1 interpreter.
  tensorflow::thread::ThreadPool* ThreadPool() { return thread_pool_.get(); }

  // Returns whether the pool is configured to use batch parallelism.
  const bool UseBatchParallelism() { return use_batch_parallelism_; }

 private:
  TfLiteInterpreterPool(
      int id,
      std::vector<std::unique_ptr<TfLiteInterpreterWrapper>> interpreters,
      std::unique_ptr<tensorflow::thread::ThreadPool> thread_pool,
      int num_interpreters, int fixed_batch_size, bool use_batch_parallelism);
  int id_;
  // A vector to maintain pointers of available interpreters
  std::vector<std::unique_ptr<TfLiteInterpreterWrapper>> interpreters_;
  std::unique_ptr<tensorflow::thread::ThreadPool> thread_pool_;
  int num_interpreters_;
  int fixed_batch_size_;
  bool use_batch_parallelism_ = false;
};

// A TFLite interpreter wrapper class which automatically returns used
// interpreter object to TfLiteInterpreterPool.
class TfLiteInterpreterWrapper {
 public:
  // Constructor for wrapper takes only an initialized interpreter.
  TfLiteInterpreterWrapper(std::unique_ptr<tflite::Interpreter> interpreter);

  // Returns the underlying interpreter.
  tflite::Interpreter* Get() { return interpreter_.get(); }

  // If using parallelism, get the allocated batch size of the interpreter.
  int GetMiniBatchSize() { return mini_batch_size_; }

  // Set the batch size.
  void SetMiniBatchSize(int mini_batch_size) {
    mini_batch_size_ = mini_batch_size;
  }

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
  tensorflow::Status SetStringData(
      const gtl::ArraySlice<tensorflow::tstring>& batch,
      TfLiteTensor* tflite_tensor, int tensor_index);

 private:
  std::unique_ptr<tflite::Interpreter> interpreter_;
  int mini_batch_size_ = 1;
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

// Contains a vector of TfLiteInterpreterPool, which are protected by mutex.
// When GetInterpreterPool is called, will either release a unique ptr to the
// caller or block if the vector is empty.
class TfLiteSessionPool {
 public:
  // Creates a TfLiteSessionPool with model, session options,
  // whether to run in caller thread, pool_size and batch_pool_size for
  // each InterpreterPool.
  static tensorflow::Status CreateTfLiteSessionPool(
      const tflite::FlatBufferModel* model,
      const tensorflow::SessionOptions& options, bool run_in_caller,
      int pool_size, int batch_pool_size,
      std::unique_ptr<TfLiteSessionPool>& tflite_session_pool);

  // Returns a TFLite interpreter pool object. Caller may *block* waiting for
  // a free interpreter pool to be available.
  std::unique_ptr<TfLiteInterpreterPool> GetInterpreterPool() {
    auto interpreter_available = [this]() ABSL_SHARED_LOCKS_REQUIRED(mutex_) {
      return !this->available_.empty();
    };
    mutex_.LockWhen(absl::Condition(&interpreter_available));
    auto pool = std::move(available_.back());
    available_.pop_back();
    mutex_.Unlock();
    return pool;
  }

  // Returns an interpreter pool to the available pool.
  void ReturnInterpreterPool(std::unique_ptr<TfLiteInterpreterPool> pool) {
    absl::MutexLock l(&mutex_);
    available_.emplace_back(std::move(pool));
  }

 private:
  TfLiteSessionPool(std::vector<std::unique_ptr<TfLiteInterpreterPool>> pools)
      : available_(std::move(pools)) {}
  mutable absl::Mutex mutex_;
  std::vector<std::unique_ptr<TfLiteInterpreterPool>> available_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace internal
}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_TFLITE_INTERPRETER_POOL_H_
