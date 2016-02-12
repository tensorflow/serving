/* Copyright 2016 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_SERVING_CORE_TARGET_H_
#define TENSORFLOW_SERVING_CORE_TARGET_H_

#include <algorithm>
#include <memory>
#include <vector>

#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow_serving/core/servable_data.h"
#include "tensorflow_serving/core/source.h"
#include "tensorflow_serving/util/observer.h"

namespace tensorflow {
namespace serving {

// An abstraction for a module that receives instructions on servables to load,
// from a Source. See source.h for documentation.
template <typename T>
class Target {
 public:
  virtual ~Target() = default;

  // Supplies a callback for a Source to use to supply aspired versions. See
  // Source<T>::AspiredVersionsCallback for the semantics of aspired versions.
  //
  // The returned function satisfies these properties:
  //  - It is thread-safe.
  //  - It is valid forever, even after this Target object has been destroyed.
  //    After this Target is gone, the function becomes a no-op.
  //  - It blocks until the target has been fully set up and is able to handle
  //    the incoming request.
  virtual typename Source<T>::AspiredVersionsCallback
  GetAspiredVersionsCallback() = 0;
};

// A base class for Target implementations. Takes care of ensuring that the
// emitted aspired-versions callbacks outlive the Target object. Target
// implementations should extend TargetBase.
template <typename T>
class TargetBase : public Target<T> {
 public:
  TargetBase();
  ~TargetBase() override = default;

  typename Source<T>::AspiredVersionsCallback GetAspiredVersionsCallback()
      final;

 protected:
  // A method supplied by the implementing subclass to handle incoming aspired-
  // versions requests from sources.
  //
  // Must be thread-safe, to handle the case of multiple sources (or one multi-
  // threaded source).
  //
  // May block until the target has been fully set up and is able to handle the
  // incoming request.
  virtual void SetAspiredVersions(const StringPiece servable_name,
                                  std::vector<ServableData<T>> versions) = 0;

 private:
  std::unique_ptr<Observer<const StringPiece, std::vector<ServableData<T>>>>
      observer_;
};

// Connects a source to a target, s.t. the target will receive the source's
// aspired-versions requests.
template <typename T>
void ConnectSourceToTarget(Source<T>* source, Target<T>* target);

//////////
// Implementation details follow. API users need not read.

template <typename T>
TargetBase<T>::TargetBase() {
  observer_.reset(new Observer<const StringPiece, std::vector<ServableData<T>>>(
      [this](const StringPiece servable_name,
             std::vector<ServableData<T>> versions) {
        this->SetAspiredVersions(servable_name, std::move(versions));
      }));
}

template <typename T>
typename Source<T>::AspiredVersionsCallback
TargetBase<T>::GetAspiredVersionsCallback() {
  return observer_->Notifier();
}

template <typename T>
void ConnectSourceToTarget(Source<T>* source, Target<T>* target) {
  source->SetAspiredVersionsCallback(target->GetAspiredVersionsCallback());
}

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_TARGET_H_
