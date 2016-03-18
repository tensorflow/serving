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

// A Unique Ptr like class that manages the lifecycle of an any dependencies in
// addition to the primary owned object.
//
// This enables the use-case of returning ownership of an object to some client
// code without giving the client access to the dependencies, and automatically
// handling destruction upon destruction of the primary owned object.
#ifndef TENSORFLOW_SERVING_UTIL_UNIQUE_PTR_WITH_DEPS_H_
#define TENSORFLOW_SERVING_UTIL_UNIQUE_PTR_WITH_DEPS_H_

#include <algorithm>
#include <memory>
#include <vector>

#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_serving/util/any_ptr.h"

namespace tensorflow {
namespace serving {

// Holds an object with its dependencies. On destruction deletes the main
// object and all dependencies, in the order inverse to the order of
// AddDependency/SetOwned calls.
template<typename T>
class UniquePtrWithDeps {
 public:
  UniquePtrWithDeps() {}
  explicit UniquePtrWithDeps(std::unique_ptr<T> object) {
    SetOwned(std::move(object));
  }
  explicit UniquePtrWithDeps(T* owned_object) { SetOwnedPtr(owned_object); }
  UniquePtrWithDeps(UniquePtrWithDeps&& other) = default;

  ~UniquePtrWithDeps() {
    // Delete all dependencies, starting with the one added last. Order of
    // destructing elements in vector/list is unspecified. The ownership of the
    // main object is kept as one of the dependencies.
    while (!deleters_.empty()) {
      deleters_.pop_back();
    }
  }

  template <typename X>
  X* AddDependency(std::unique_ptr<X> dependency) {
    X* raw = dependency.get();
    deleters_.emplace_back(std::move(dependency));
    return raw;
  }

  void SetOwned(std::unique_ptr<T> object) {
    object_ = AddDependency<T>(std::move(object));
  }
  void SetOwnedPtr(T* owned_object) {
    SetOwned(std::unique_ptr<T>(owned_object));
  }

  T* get() const { return object_; }
  const T& operator*() const { return *object_; }
  T* operator->() const { return get(); }

 private:
  std::vector<UniqueAnyPtr> deleters_;
  T* object_ = nullptr;
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_UTIL_UNIQUE_PTR_WITH_DEPS_H_
