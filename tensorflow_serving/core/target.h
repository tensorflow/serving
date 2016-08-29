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

#include "tensorflow/core/lib/core/notification.h"
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
//
// IMPORTANT: Every leaf derived class must call Detach() at the top of its
// destructor. (See documentation on Detach() below.)
template <typename T>
class TargetBase : public Target<T> {
 public:
  ~TargetBase() override;

  typename Source<T>::AspiredVersionsCallback GetAspiredVersionsCallback()
      final;

 protected:
  // This is an abstract class.
  TargetBase();

  // A method supplied by the implementing subclass to handle incoming aspired-
  // versions requests from sources.
  //
  // IMPORTANT: The SetAspiredVersions() implementation must be thread-safe, to
  // handle the case of multiple sources (or one multi-threaded source).
  //
  // May block until the target has been fully set up and is able to handle the
  // incoming request.
  virtual void SetAspiredVersions(const StringPiece servable_name,
                                  std::vector<ServableData<T>> versions) = 0;

  // Stops receiving SetAspiredVersions() calls. Every leaf derived class (i.e.
  // sub-sub-...-class with no children) must call Detach() at the top of its
  // destructor to avoid races with state destruction. After Detach() returns,
  // it is guaranteed that no SetAspiredVersions() calls are running (in any
  // thread) and no new ones can run. Detach() must be called exactly once.
  void Detach();

 private:
  // Used to synchronize all class state. The shared pointer permits use in an
  // observer lambda while being impervious to this class's destruction.
  mutable std::shared_ptr<mutex> mu_;

  // Notified when Detach() has been called. The shared pointer permits use in
  // an observer lambda while being impervious to this class's destruction.
  std::shared_ptr<Notification> detached_;

  // An observer object that forwards to SetAspiredVersions(), if not detached.
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
TargetBase<T>::TargetBase() : mu_(new mutex), detached_(new Notification) {
  std::shared_ptr<mutex> mu = mu_;
  std::shared_ptr<Notification> detached = detached_;
  observer_.reset(new Observer<const StringPiece, std::vector<ServableData<T>>>(
      [mu, detached, this](const StringPiece servable_name,
                           std::vector<ServableData<T>> versions) {
        mutex_lock l(*mu);
        if (detached->HasBeenNotified()) {
          // We're detached. Perform a no-op.
          return;
        }
        this->SetAspiredVersions(servable_name, std::move(versions));
      }));
}

template <typename T>
TargetBase<T>::~TargetBase() {
  DCHECK(detached_->HasBeenNotified()) << "Detach() must be called exactly "
                                          "once, at the top of the leaf "
                                          "derived class's destructor";
}

template <typename T>
typename Source<T>::AspiredVersionsCallback
TargetBase<T>::GetAspiredVersionsCallback() {
  mutex_lock l(*mu_);
  if (detached_->HasBeenNotified()) {
    // We're detached. Return a no-op callback.
    return [](const StringPiece, std::vector<ServableData<T>>) {};
  }
  return observer_->Notifier();
}

template <typename T>
void TargetBase<T>::Detach() {
  DCHECK(!detached_->HasBeenNotified()) << "Detach() must be called exactly "
                                           "once, at the top of the leaf "
                                           "derived class's destructor";

  // We defer deleting the observer until after we've released the lock, to
  // avoid a deadlock with the observer's internal lock when it calls our
  // lambda.
  std::unique_ptr<Observer<const StringPiece, std::vector<ServableData<T>>>>
      detached_observer;
  {
    mutex_lock l(*mu_);
    detached_observer = std::move(observer_);
    if (!detached_->HasBeenNotified()) {
      detached_->Notify();
    }
  }
}

template <typename T>
void ConnectSourceToTarget(Source<T>* source, Target<T>* target) {
  source->SetAspiredVersionsCallback(target->GetAspiredVersionsCallback());
}

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_TARGET_H_
