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

#ifndef TENSORFLOW_SERVING_CORE_SERVABLE_HANDLE_H_
#define TENSORFLOW_SERVING_CORE_SERVABLE_HANDLE_H_

#include <algorithm>
#include <cstddef>
#include <memory>
#include <type_traits>

#include "tensorflow_serving/core/loader.h"
#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/util/any_ptr.h"

namespace tensorflow {
namespace serving {

// A non-templatized handle to a servable, used internally in the
// Manager to retrieve a type-erased servable object from the Loader.
// The handle keeps the underlying object alive as long as the handle is alive.
// The frontend should not hold onto it for a long time, because holding it can
// delay servable reloading.
class UntypedServableHandle {
 public:
  virtual ~UntypedServableHandle() = default;

  virtual AnyPtr servable() = 0;
};

// A smart pointer to the underlying servable object T retrieved from the
// Loader. Frontend code gets these handles from the ServableManager. The
// handle keeps the underlying object alive as long as the handle is alive. The
// frontend should not hold onto it for a long time, because holding it can
// delay servable reloading.
//
// The T returned from the handle is generally shared among multiple requests,
// which means any mutating changes made to T must preserve correctness
// vis-a-vis the application logic. Moreover, in the presence of multiple
// request threads, thread-safe usage of T must be ensured.
//
// T is expected to be a value type, and is internally stored as a pointer.
// Using a pointer type for T will fail to compile, since it would be a mistake
// to do so in most situations.
//
// Example Use:
//  // Define or use an existing servable:
//  class MyServable {
//   public:
//    void MyMethod();
//  };
//
//  // Get your handle from a manager.
//  ServableHandle<MyServable> handle;
//  TF_RETURN_IF_ERROR(manager->GetServableHandle(id, &handle));
//
//  // Use your handle as a smart-pointer:
//  handle->MyMethod();
template <typename T>
class ServableHandle {
 public:
  static_assert(!std::is_pointer<T>::value,
                "Servables are implicitly passed as pointers, please use T "
                "instead of T*.");

  // ServableHandle is null by default.
  ServableHandle() = default;

  // Implicit cast from null.
  ServableHandle(std::nullptr_t)  // NOLINT(runtime/explicit)
      : ServableHandle() {}

  // Smart pointer operations.

  T& operator*() const { return *get(); }

  T* operator->() const { return get(); }

  T* get() const { return servable_; }

  operator bool() const { return get() != nullptr; }

  // See the end of this file for comparison operators, which must be declared
  // at namespace scope to support left-hand-side arguments of different types.

 private:
  friend class Manager;

  explicit ServableHandle(std::unique_ptr<UntypedServableHandle> untyped_handle)
      : untyped_handle_(std::move(untyped_handle)),
        servable_(untyped_handle_ == nullptr
                      ? nullptr
                      : untyped_handle_->servable().get<T>()) {}

  std::unique_ptr<UntypedServableHandle> untyped_handle_;
  T* servable_ = nullptr;
};

// An implementation of UntypedServableHandle using shared_ptr to do
// ref-counting on the Loader that owns the Servable.
class SharedPtrHandle final : public UntypedServableHandle {
 public:
  ~SharedPtrHandle() override = default;

  explicit SharedPtrHandle(std::shared_ptr<Loader> loader)
      : loader_(std::move(loader)) {}

  AnyPtr servable() override { return loader_->servable(); }

 private:
  std::shared_ptr<Loader> loader_;
};

// Macro to define relational operators for ServableHandle without too much
// boiler-plate.
//
// Note that these are not deep comparisons, only the addresses are used.
#define SERVABLE_HANDLE_REL_OP(OP)                                         \
  template <typename T, typename U>                                        \
  constexpr bool operator OP(const ServableHandle<T>& l,                   \
                             const ServableHandle<U>& r) {                 \
    return l.get() OP r.get();                                             \
  }                                                                        \
  template <typename T>                                                    \
  constexpr bool operator OP(std::nullptr_t, const ServableHandle<T>& r) { \
    return nullptr OP r.get();                                             \
  }                                                                        \
  template <typename T>                                                    \
  constexpr bool operator OP(const ServableHandle<T>& l, std::nullptr_t) { \
    return l.get() OP nullptr;                                             \
  }

SERVABLE_HANDLE_REL_OP(==)
SERVABLE_HANDLE_REL_OP(!=)

#undef SERVABLE_HANDLE_REL_OP
}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_SERVABLE_HANDLE_H_
