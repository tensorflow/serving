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
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/lite/model.h"

namespace tensorflow {
namespace serving {
namespace internal {

class TfLiteInterpreterWrapper;

// A TFLite interpreter pool to maintain pre-created tflite::Interpreter
// objects. This class is thread-safe.
class TfLiteInterpreterPool {
 public:
  // Create an interpreter pool with the given TFLite model and number of
  // interpreters. The model must outlive the returned pool instance.
  static Status CreateTfLiteInterpreterPool(
      const tflite::FlatBufferModel& model, int num_interpreters,
      std::unique_ptr<TfLiteInterpreterPool>& pool);
  // Returns a TFLite interpreter wrapper object. Caller may *block* waiting for
  // a free interpreter to be available.
  std::unique_ptr<TfLiteInterpreterWrapper> GetInterpreter();

 private:
  TfLiteInterpreterPool(
      std::vector<std::unique_ptr<tflite::Interpreter>> interpreters);
  void ReturnInterpreter(tflite::Interpreter* interpreter);
  mutable absl::Mutex mutex_;
  // A vector to maintain pointers of available interpreters
  std::vector<tflite::Interpreter*> available_ ABSL_GUARDED_BY(mutex_);
  const std::vector<std::unique_ptr<tflite::Interpreter>> interpreters_;

  friend TfLiteInterpreterWrapper;
};

// A TFLite interpreter wrapper class which automatically returns used
// interpreter object to TfLiteInterpreterPool.
class TfLiteInterpreterWrapper {
 public:
  TfLiteInterpreterWrapper(tflite::Interpreter* interpreter,
                           TfLiteInterpreterPool* pool)
      : interpreter_(interpreter), pool_(pool) {}

  ~TfLiteInterpreterWrapper() { pool_->ReturnInterpreter(interpreter_); }
  tflite::Interpreter* Get() { return interpreter_; }

 private:
  tflite::Interpreter* const interpreter_;
  TfLiteInterpreterPool* const pool_;
};

}  // namespace internal
}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_TFLITE_INTERPRETER_POOL_H_
