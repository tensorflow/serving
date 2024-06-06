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
#include <atomic>
#include <functional>
#include <memory>
#include <string>

#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/random.h"
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

// FastReadDynamicPtr, below, is templatized both on the type pointed to, and on
// a concept named "ReadPtrHolder", which holds the ReadPtrs for the
// FastReadDynamicPtr instance.  While we could hold the ReadPtr with just a
// single ReadPtr and a mutex (see SingleReadPtr, below) having a separate
// concept lets us use higher performance mechanisms like sharding the ReadPtrs
// to reduce contention.
namespace internal_read_ptr_holder {
template <typename T>
class ShardedReadPtrs;
}  // namespace internal_read_ptr_holder

template <typename T,
          typename ReadPtrHolder = internal_read_ptr_holder::ShardedReadPtrs<T>>
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
  ~FastReadDynamicPtr();

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
  // ShareableOwnedPtr wraps an OwnedPtr with an interface that can provide
  // multiple independent ReadPtrs to it.  These ReadPtrs will have separate
  // reference counts to reduce contention.
  class ShareableOwnedPtr;

  mutex mu_;
  std::unique_ptr<ShareableOwnedPtr> shareable_;

  ReadPtrHolder read_ptrs_;

  TF_DISALLOW_COPY_AND_ASSIGN(FastReadDynamicPtr);
};

template <typename T, typename ReadPtrHolder>
class FastReadDynamicPtr<T, ReadPtrHolder>::ShareableOwnedPtr {
 public:
  explicit ShareableOwnedPtr(OwnedPtr p) : owned_(std::move(p)) {}

  // Returns a new, independent ReadPtr referencing the held OwnedPtr.
  // Release() will not return the OwnedPtr until the returned ReadPtr and all
  // its copies are destroyed.
  ReadPtr NewShare() {
    if (owned_ == nullptr) {
      return nullptr;
    }
    shares_.fetch_add(1, std::memory_order_release);
    return std::shared_ptr<T>(owned_.get(), [this](T* p) { DecRef(); });
  }

  // Waits until all shares have been destroyed, and then returns the OwnedPtr.
  // No methods should be called after this one.
  OwnedPtr Release() && {
    DecRef();
    no_longer_shared_.WaitForNotification();
    return std::move(owned_);
  }

 private:
  void DecRef() {
    if (shares_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      no_longer_shared_.Notify();
    }
  }

  OwnedPtr owned_;
  // The number of times we've shared the pointer.  This defaults to 1 so we can
  // safely and incrementally hand out shares, without worrying that
  // no_longer_shared_ will be notified until Release() has been called, which
  // decrements this before waiting for a notification.
  std::atomic<uint32> shares_ = {1};
  // When shares_ goes to zero, this will be notified.
  Notification no_longer_shared_;
  TF_DISALLOW_COPY_AND_ASSIGN(ShareableOwnedPtr);
};

namespace internal_read_ptr_holder {
// ReadPtrHolders must provide two methods:
//   1. get(), which must be thread safe, and returns a shared_ptr<const T>, and
//   2. update(), which takes a factory function f (which returns a
//      std::shared_ptr<const T>), calls it one or more times, and updates the
//      internal state to match.  get() after an update() call should return one
//      of the pointers produced by the factory.  update() is not required to be
//      thread-safe against other callers (but must be thread-safe against
//      parallel get() calls); it will only ever be called under a lock.
//
// The Factory-based interface of update() may seem strange, but it allows
// ReadPtrHolders to hold several distinct ReadPtrs.

// SingleReadDPtr is the simplest possible implementation of a ReadPtrHolder,
// but it causes every reader to contend on both the mutex_lock and the atomic
// reference count for the ReadPtr it holds.  By default we use ShardedReadPtrs,
// below, which avoids both of these pitfalls.  SingleReadPtr here is useful
// primarily for benchmarking and for ensuring that no reader ever goes "back in
// time": in the sharded implementation, below, it's possible for a reader to
// see a new version and then see an older version in get(), if the writer is in
// the midst of an update() call.
//
// If you don't care about contention and you want to save memory, you might
// want to use this.
template <typename T>
class SingleReadPtr {
 public:
  std::shared_ptr<const T> get() const {
    mutex_lock lock(mu_);
    return p_;
  }

  template <typename Factory>
  void update(const Factory& f) {
    auto p = f();
    mutex_lock lock(mu_);
    p_.swap(p);
  }

 private:
  mutable mutex mu_;
  std::shared_ptr<const T> p_;
};

// This maintains a set of sharded ReadPtrs.  It tries to shard one ReadPtr per
// CPU, but if the port::NumTotalCPUs or port::GetCurrentCPU fails, it falls
// back to random sharding.
template <typename T>
class ShardedReadPtrs {
 public:
  ShardedReadPtrs() : shards_(new PaddedThreadSafeSharedPtr[num_shards_]) {}

  std::shared_ptr<const T> get() const {
    const int shard = GetShard();
    mutex_lock lock(shards_[shard].mu);
    return shards_[shard].ps[index_.load(std::memory_order_acquire)];
  }

  template <typename Factory>
  void update(const Factory& f) {
    // First we'll update all the pointers into each shard's next_index, then
    // we'll change index_ to point to those new pointers, then we'll get rid of
    // orig_index.  This ensures temporal consistency, so no reader ever goes
    // back in time: all readers advance to next_index together, when we write
    // to index_.
    const uint32 orig_index = index_.load(std::memory_order_acquire);
    const uint32 next_index = orig_index ? 0 : 1;
    for (int shard = 0; shard < num_shards_; ++shard) {
      auto p = f();
      mutex_lock lock(shards_[shard].mu);
      shards_[shard].ps[next_index] = std::move(p);
    }
    index_.store(next_index, std::memory_order_release);
    for (int shard = 0; shard < num_shards_; ++shard) {
      std::shared_ptr<const T> p;
      mutex_lock lock(shards_[shard].mu);
      shards_[shard].ps[orig_index].swap(p);
    }
  }

 private:
  // NOTE: If a std::atomic_shared_ptr is ever available, it would be reasonable
  // to use that here for improved performance.
  struct ThreadSafeSharedPtr {
    std::shared_ptr<const T> ps[2];
    mutex mu;
  };

  // We pad the pointers to ensure that individual shards don't experience false
  // sharing between threads.
  struct PaddedThreadSafeSharedPtr : public ThreadSafeSharedPtr {
    char padding[64 - sizeof(ThreadSafeSharedPtr)];
  };
  static_assert(sizeof(PaddedThreadSafeSharedPtr) >= 64,
                "PaddedThreadSafeSharedPtr should be at least 64 bytes.");

  static constexpr int kRandomShards = 16;
  int GetShard() const {
    const int cpu = port::GetCurrentCPU();
    if (cpu != -1) {
      return cpu;
    }
    // Otherwise, return a random shard.  random::New64 would introduce a mutex
    // lock here, which would defeat the purpose of the sharding.  Similarly, a
    // static std::atomic<uint64_t>, if updated with any memory order other than
    // std::memory_order_relaxed, would re-introduce contention on that memory
    // location.  A thread_local sidesteps both problems with only eight bytes
    // per thread of overhead.
    //
    // MCGs need to be seeded with an odd number, so we ensure the lowest bit is
    // set.
    thread_local uint64_t state = {random::New64() | 1ULL};
    // We just need something simple and good enough.  The multiplier here was
    // picked from "COMPUTATIONALLY EASY, SPECTRALLY GOOD MULTIPLIERS FOR
    // CONGRUENTIAL PSEUDORANDOM NUMBER GENERATORS" by Steele and Vigna.
    state *= 0xd09d;
    // Update this shift if kRandomShards changes.
    return state >> 60;
  }

 protected:
  const int num_shards_ =
      port::NumTotalCPUs() == -1 ? kRandomShards : port::NumTotalCPUs();
  std::atomic<uint32> index_{0};
  std::unique_ptr<PaddedThreadSafeSharedPtr[]> shards_;
};

}  // namespace internal_read_ptr_holder

template <typename T, typename ReadPtrHolder>
FastReadDynamicPtr<T, ReadPtrHolder>::FastReadDynamicPtr(OwnedPtr p) {
  if (p != nullptr) {
    Update(std::move(p));
  }
}

template <typename T, typename ReadPtrHolder>
FastReadDynamicPtr<T, ReadPtrHolder>::~FastReadDynamicPtr() {
  // Force a wait until all outstanding ReadPtrs are destroyed.
  Update(nullptr);
}

template <typename T, typename ReadPtrHolder>
std::unique_ptr<T> FastReadDynamicPtr<T, ReadPtrHolder>::Update(
    OwnedPtr new_object) {
  std::unique_ptr<ShareableOwnedPtr> shareable(
      new ShareableOwnedPtr(std::move(new_object)));
  {
    mutex_lock lock(mu_);
    read_ptrs_.update([&] { return shareable->NewShare(); });
    shareable_.swap(shareable);
  }
  return shareable == nullptr ? nullptr : std::move(*shareable).Release();
}

template <typename T, typename ReadPtrHolder>
std::shared_ptr<const T> FastReadDynamicPtr<T, ReadPtrHolder>::get() const {
  return read_ptrs_.get();
}

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_UTIL_FAST_READ_DYNAMIC_PTR_H_
