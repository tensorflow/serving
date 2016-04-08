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

#include "tensorflow_serving/core/manager.h"

#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow_serving/util/any_ptr.h"

namespace tensorflow {
namespace serving {
namespace {

struct TestServable {
  int member = 7;
};

class TestHandle : public UntypedServableHandle {
 public:
  AnyPtr servable() override { return &servable_; }

 private:
  TestServable servable_;
};

// A manager that a returns a TestHandle.
class TestManager : public Manager {
 public:
  std::vector<ServableId> ListAvailableServableIds() override {
    LOG(FATAL) << "Not expected to be called.";
  }

 private:
  Status GetUntypedServableHandle(
      const ServableRequest& request,
      std::unique_ptr<UntypedServableHandle>* result) override {
    result->reset(new TestHandle);
    return Status::OK();
  }

  std::map<ServableId, std::unique_ptr<UntypedServableHandle>>
  GetAvailableUntypedServableHandles() const override {
    std::map<ServableId, std::unique_ptr<UntypedServableHandle>> handles;
    handles.emplace(ServableId{"Foo", 2},
                    std::unique_ptr<UntypedServableHandle>(new TestHandle));
    return handles;
  }
};

TEST(ManagerTest, NoErrors) {
  TestManager manager;
  ServableHandle<TestServable> handle;
  EXPECT_TRUE(manager.GetServableHandle({"Foo", 2}, &handle).ok());
  EXPECT_NE(nullptr, handle.get());
}

TEST(ManagerTest, TypeError) {
  TestManager manager;
  ServableHandle<int> handle;
  EXPECT_FALSE(manager.GetServableHandle({"Foo", 2}, &handle).ok());
  EXPECT_EQ(nullptr, handle.get());
}

TEST(ManagerTest, GetAvailableServableHandles) {
  TestManager manager;
  const std::map<ServableId, ServableHandle<TestServable>> handles =
      manager.GetAvailableServableHandles<TestServable>();
  ASSERT_EQ(1, handles.size());
  for (const auto& handle : handles) {
    EXPECT_EQ((ServableId{"Foo", 2}), handle.first);
    EXPECT_EQ(7, handle.second->member);
  }
}

TEST(ManagerTest, GetAvailableServableHandlesWrongType) {
  TestManager manager;
  const std::map<ServableId, ServableHandle<int>> handles =
      manager.GetAvailableServableHandles<int>();
  EXPECT_EQ(0, handles.size());
}

// A manager that returns OK even though the result is null. This behavior
// violates the interface of Manager, but it is used to test that this violation
// is handled gracefully rather than a crash or memory corruption.
class ReturnNullManager : public TestManager {
 private:
  Status GetUntypedServableHandle(
      const ServableRequest& request,
      std::unique_ptr<UntypedServableHandle>* result) override {
    *result = nullptr;
    return Status::OK();
  }
};

TEST(ManagerTest, NullHandleReturnsError) {
  ReturnNullManager manager;
  ServableHandle<TestServable> handle;
  EXPECT_FALSE(manager.GetServableHandle({"Foo", 2}, &handle).ok());
  EXPECT_EQ(nullptr, handle.get());
}

// A manager that returns an error even though the result is non-null.
class ReturnErrorManager : public TestManager {
 private:
  Status GetUntypedServableHandle(
      const ServableRequest& request,
      std::unique_ptr<UntypedServableHandle>* result) override {
    result->reset(new TestHandle);
    return errors::Internal("Something bad happened.");
  }
};

TEST(ManagerTest, ErrorReturnsNullHandle) {
  ReturnErrorManager manager;
  ServableHandle<TestServable> handle;
  EXPECT_FALSE(manager.GetServableHandle({"Foo", 2}, &handle).ok());
  EXPECT_EQ(nullptr, handle.get());
}

// Wraps a pointer as a servable. Does a bit of type-gymnastics since servable
// handles can only be constructed by managers.
template <typename T>
ServableHandle<T> WrapAsHandle(T* t) {
  // Perform some type gymnastics to create a handle that points to 't'.
  class DummyHandle : public UntypedServableHandle {
   public:
    explicit DummyHandle(T* servable) : servable_(servable) {}

    AnyPtr servable() override { return servable_; }

   private:
    T* servable_;
  };

  // Always returns the same servable.
  class DummyManager : public TestManager {
   public:
    explicit DummyManager(T* servable) : servable_(servable) {}

    Status GetUntypedServableHandle(
        const ServableRequest& request,
        std::unique_ptr<UntypedServableHandle>* result) override {
      result->reset(new DummyHandle(servable_));
      return Status::OK();
    }

   private:
    T* servable_;
  };

  DummyManager manager{t};
  ServableHandle<T> handle;
  TF_CHECK_OK(manager.GetServableHandle({"Dummy", 0}, &handle));
  return handle;
}

TEST(ServableHandleTest, PointerOps) {
  TestServable servables[2];
  ServableHandle<TestServable> handles[2];

  handles[0] = WrapAsHandle(&servables[0]);
  handles[1] = WrapAsHandle(&servables[1]);
  ServableHandle<TestServable> null_handle = nullptr;

  // Equality.
  EXPECT_EQ(handles[0], handles[0]);
  EXPECT_EQ(null_handle, nullptr);
  EXPECT_EQ(nullptr, null_handle);

  // Inequality.
  EXPECT_NE(handles[0], handles[1]);
  EXPECT_NE(handles[0], nullptr);
  EXPECT_NE(nullptr, handles[0]);

  // Bool conversion.
  EXPECT_TRUE(handles[0]);
  EXPECT_FALSE(null_handle);

  // Dereference and get.
  EXPECT_EQ(&servables[0], handles[0].get());
  EXPECT_EQ(&servables[0], &*handles[0]);
  EXPECT_EQ(&servables[0].member, &handles[0]->member);
}

TEST(ServableRequestTest, Specific) {
  const auto request = ServableRequest::Specific("servable", 7);
  EXPECT_EQ("servable", request.name);
  EXPECT_EQ(7, *request.version);
}

TEST(ServableRequestTest, Latest) {
  const auto request = ServableRequest::Latest("servable");
  EXPECT_EQ("servable", request.name);
  EXPECT_FALSE(request.version);
}

TEST(ServableRequestTest, FromId) {
  const auto request = ServableRequest::FromId({"servable", 7});
  EXPECT_EQ("servable", request.name);
  EXPECT_EQ(7, *request.version);
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
