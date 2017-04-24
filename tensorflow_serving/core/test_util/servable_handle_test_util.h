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

#ifndef TENSORFLOW_SERVING_CORE_TEST_UTIL_SERVABLE_HANDLE_TEST_UTIL_H_
#define TENSORFLOW_SERVING_CORE_TEST_UTIL_SERVABLE_HANDLE_TEST_UTIL_H_

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_serving/core/manager.h"
#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/core/servable_id.h"

namespace tensorflow {
namespace serving {
namespace test_util {

// Wraps a pointer as a servable.
//
// The ownership of @p t is *not* transferred to this function; t must outlive
// the returned ServableHandle.
//
// Does a bit of type-gymnastics since ServableHandles can only be constructed
// by Managers.
template <typename T>
static ServableHandle<T> WrapAsHandle(const ServableId& id, T* t) {
  // A basic Handle that wraps a ServableId and a pointer to a servable.
  //
  // Exactly the same objects that are used to construct the DummyHandle are
  // returned when the appropriate getter functions are invoked.
  class DummyHandle : public UntypedServableHandle {
   public:
    explicit DummyHandle(const ServableId& id, T* servable)
        : id_(id), servable_(servable) {}

    AnyPtr servable() override { return servable_; }

    const ServableId& id() const override { return id_; }

   private:
    const ServableId id_;
    T* servable_;
  };

  // A Manager that always returns the same servable when
  // GetUntypedServableHandle is invoked.
  class DummyManager : public Manager {
   public:
    explicit DummyManager(const ServableId& id, T* servable)
        : id_(id), servable_(servable) {}

    // Resets the UntypedServableHandle to a new DummyHandle that wraps the
    // Servable and ServableId which this DummyManager was created with.
    //
    // Always returns OK status.
    Status GetUntypedServableHandle(
        const ServableRequest& request,
        std::unique_ptr<UntypedServableHandle>* result) override {
      result->reset(new DummyHandle(id_, servable_));
      return Status::OK();
    }

    // Unimplemented: always returns an empty map.
    std::map<ServableId, std::unique_ptr<UntypedServableHandle>>
    GetAvailableUntypedServableHandles() const override {
      return {};
    }

    // Unimplemented: always returns an empty vector.
    std::vector<ServableId> ListAvailableServableIds() const override {
      return {};
    }

   private:
    const ServableId id_;
    T* servable_;
  };

  DummyManager manager{id, t};
  ServableHandle<T> handle;
  TF_CHECK_OK(manager.GetServableHandle({"Dummy", 0}, &handle));
  return handle;
}

}  // namespace test_util
}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_TEST_UTIL_SERVABLE_HANDLE_TEST_UTIL_H_
