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

#ifndef TENSORFLOW_SERVING_UTIL_FAST_READ_DYNAMIC_PTR_H_
#define TENSORFLOW_SERVING_UTIL_FAST_READ_DYNAMIC_PTR_H_

#include <algorithm>
#include <functional>
#include <memory>
#include <string>

#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace serving {

// FastReadDynamicPtr<> is a thread-safe class used to manage an object that
// needs to be updated occasionally while being accessed from multiple threads.
//
// Typical use requires construction of a new object while the old object is
// still being used, calling Update() when the new object is ready.  Access to
// the object is asynchronous relative to update operations, so pointers may
// exist to both the old and the new object at the same time (although there can
// never be more than two objects concurrently). After the update begins, any
// new calls to get() will point to the new object. The update will then block
// until all pointers to the old object go out of scope.  This is achieved via a
// reference counted smart pointer.
//
// This class is functionally very similar to using a shared_ptr guarded by a
// mutex, with the important distinction that it provides finer control over
// which thread destroys the object, the number of live objects, the ability to
// recycle old objects if desired, and it forces an efficient pattern for
// updating data (swapping in a pointer rather than in-place modification).
//
// Example Use:
//
//  In initialization code:
//    FastReadDynamicPtr<HeavyWeightObject> ptr{InitialValue()};
//
//  From updating thread (executes infrequently, not performance sensitive):
//    std::unique_ptr<HeavyWeightObject> new_object(LongRunningFunction());
//    std::unique_ptr<HeavyWeightObject> old_object =
//        ptr.Update(std::move(new_object));
//    // Reuse old_object, or just destroy it.
//
//  From reading thread (executes frequently, high performance requirements):
//    auto object = ptr.get();
//    if (object != nullptr) {
//      HandleRequest(object.get());
//    }
//
// Care must be taken to not call FastReadDynamicPtr::Update() from a thread
// that owns any instances of FastReadDynamicPtr::ReadPtr, or else deadlock may
// occur.
template <typename T>
class FastReadDynamicPtr {
 public:
  // Short, documentative names for the types of smart pointers we use. Callers
  // are not required to use these names; shared_ptr and unique_ptr are part of
  // the interface. This is particularly useful for calling the aliased pointer
  // constructor of shared_ptr.

  // Used when providing a read-only pointer. Never actually used to own an
  // object.
  using ReadPtr = std::shared_ptr<const T>;

  // Used when an object is owned.
  using OwnedPtr = std::unique_ptr<T>;

  // Initially contains a null pointer by default.
  explicit FastReadDynamicPtr(OwnedPtr = nullptr);

  // Updates the current object with a new one, returning the old object. This
  // method will block until all ReadPtrs that point to the previous object have
  // been destroyed, guaranteeing that the result is truly unique upon return.
  // This method may be called with a null pointer.
  //
  // If the current thread owns any ReadPtrs to the current object, this method
  // will deadlock.
  OwnedPtr Update(OwnedPtr new_object);

  // Returns a read-only pointer to the current object. The object will not be
  // invalidated as long as the returned ReadPtr hasn't been destroyed. The
  // return value may be null if update hasn't been called and if no initial
  // value is provided.
  //
  // Note that Update() should not be called from this thread while the returned
  // ReadPtr is in scope, or deadlock will occur.
  ReadPtr get() const;

 private:
  // A class that behaves like a shared_ptr, except it is capable of being
  // released (as a unique_ptr) when it becomes unique.
  class ReleasableSharedPtr;

  // The current pointer, and a mutex to guard it. Note that the only operations
  // performed under lock are swap, during Update(), and incrementing the
  // reference count, during get().
  // TODO(b/24973960): Consider implementing userspace RCU instead of this if
  // performance is ever a concern.
  mutable mutex mutex_;
  std::unique_ptr<ReleasableSharedPtr> object_;

  TF_DISALLOW_COPY_AND_ASSIGN(FastReadDynamicPtr);
};

//
// Implementation details follow.
//

template <typename T>
class FastReadDynamicPtr<T>::ReleasableSharedPtr {
 public:
  explicit ReleasableSharedPtr(OwnedPtr object)
      : object_{std::move(object)},
        read_only_object_{
            object_.get(),
            // Use a destructor that will notify 'no_longer_referenced_' rather
            // than deleting.
            std::bind(&Notification::Notify, &no_longer_referenced_)} {}

  ~ReleasableSharedPtr() {
    // Block destruction until all outstanding references have been cleaned up.
    // This prevents the last shared_ptr from calling
    // no_longer_referenced_.set_value() after destruction.
    BlockingRelease();
  }

  // Returns a reference to the underlying object, increasing the reference
  // count by one. May be called concurrently from different threads, but not
  // concurrently with BlockingRelease().
  ReadPtr reference() const { return read_only_object_; }

  // Blocks until outstanding values returned by 'reference' have been
  // destroyed.  Requires that reference() is not being called concurrently.
  OwnedPtr BlockingRelease() {
    // Allow the reference count to go to zero.
    read_only_object_ = nullptr;

    // shared_ptr doesn't call the destructor if it is null, so do not block for
    // null pointers.
    if (object_ != nullptr) {
      no_longer_referenced_.WaitForNotification();
    }

    // Yield ownership to the caller.
    return std::move(object_);
  }

 private:
  // The current object.
  OwnedPtr object_;

  // Notified when read_only_object_'s reference count goes to zero.
  Notification no_longer_referenced_;

  // A shared pointer to object_. Does not actually delete the pointer,
  // but satisifies 'no_longer_referenced_' upon destruction.
  ReadPtr read_only_object_;

  TF_DISALLOW_COPY_AND_ASSIGN(ReleasableSharedPtr);
};

template <typename T>
FastReadDynamicPtr<T>::FastReadDynamicPtr(OwnedPtr ptr)
    : object_{new ReleasableSharedPtr{std::move(ptr)}} {}

template <typename T>
std::unique_ptr<T> FastReadDynamicPtr<T>::Update(std::unique_ptr<T> object) {
  // Construct a ReleasableSharedPtr outside of the lock, this performs about
  // three allocations (the ReleasableSharedPtr, the shared_ptr control block,
  // and the internal state of std::promise) so we take care to keep it out of
  // the critical section.
  std::unique_ptr<ReleasableSharedPtr> local_ptr(
      new ReleasableSharedPtr{std::move(object)});

  // Swap the new ReleasableSharedPtr under lock.
  {
    mutex_lock lock(mutex_);
    using std::swap;
    swap(object_, local_ptr);
  }

  // Now local_ptr points to the old object, release it to the caller.  This may
  // block for a while, so this also must be kept outside of the critical
  // section.
  return local_ptr->BlockingRelease();
}

template <typename T>
typename FastReadDynamicPtr<T>::ReadPtr FastReadDynamicPtr<T>::get() const {
  mutex_lock lock(mutex_);
  return object_->reference();
}

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_UTIL_FAST_READ_DYNAMIC_PTR_H_
