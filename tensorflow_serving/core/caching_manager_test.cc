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

#include "tensorflow_serving/core/caching_manager.h"

#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow_serving/core/servable_data.h"
#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/core/servable_id.h"
#include "tensorflow_serving/core/servable_state.h"
#include "tensorflow_serving/core/servable_state_monitor.h"
#include "tensorflow_serving/core/simple_loader.h"
#include "tensorflow_serving/util/event_bus.h"
#include "tensorflow_serving/util/optional.h"
#include "tensorflow_serving/util/threadpool_executor.h"

namespace tensorflow {
namespace serving {
namespace {

using ::testing::UnorderedElementsAre;
using ::testing::UnorderedElementsAreArray;

// A simple loader-factory that concatenates requested servable name and
// version.
class StringLoaderFactory : public CachingManager::LoaderFactory {
 public:
  StringLoaderFactory() = default;
  ~StringLoaderFactory() override = default;

  Status CreateLoader(
      const ServableRequest& request,
      std::unique_ptr<ServableData<std::unique_ptr<Loader>>>* loaded_data) {
    auto servable_creator = [&](std::unique_ptr<string>* servable) {
      servable->reset(new string);
      if (!request.version) {
        return errors::NotFound("servable creator error");
      }
      **servable = strings::StrCat(request.name, "-", *request.version);
      return Status::OK();
    };
    if (!request.version) {
      // The request has no version. So we build a servable-id with a static
      // version (in this case, 42).
      const ServableId id = {request.name, 42};
      loaded_data->reset(new ServableData<std::unique_ptr<Loader>>(
          id, errors::Unknown("error")));
    } else {
      std::unique_ptr<Loader> loader;
      loader.reset(new SimpleLoader<string>(
          servable_creator, SimpleLoader<string>::EstimateNoResources()));
      const ServableId id = {request.name, *request.version};
      loaded_data->reset(
          new ServableData<std::unique_ptr<Loader>>(id, std::move(loader)));
    }
    return Status::OK();
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(StringLoaderFactory);
};

// A simple loader-factory that always returns a loader with an error for every
// request.
class ErrorLoaderFactory : public CachingManager::LoaderFactory {
 public:
  ErrorLoaderFactory() = default;
  ~ErrorLoaderFactory() override = default;

  Status CreateLoader(
      const ServableRequest& request,
      std::unique_ptr<ServableData<std::unique_ptr<Loader>>>* loaded_data) {
    auto servable_creator = [&](std::unique_ptr<string>* servable) {
      return errors::Unknown("error loader-factory");
    };
    std::unique_ptr<Loader> loader;
    loader.reset(new SimpleLoader<string>(
        servable_creator, SimpleLoader<string>::EstimateNoResources()));
    const ServableId id = {request.name, *request.version};
    loaded_data->reset(
        new ServableData<std::unique_ptr<Loader>>(id, std::move(loader)));
    return Status::OK();
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ErrorLoaderFactory);
};

// Commonly used servable names.
constexpr char kServableName[] = "kServableName";
constexpr char kServableName2[] = "kServableName2";

constexpr int kNumThreads = 10;

class CachingManagerTest : public ::testing::TestWithParam<int> {
 protected:
  CachingManagerTest()
      : servable_event_bus_(EventBus<ServableState>::CreateEventBus()),
        servable_state_monitor_(servable_event_bus_.get()) {
    num_load_unload_threads_ = GetParam();
    max_num_load_retries_ = 1;

    CachingManager::Options options;
    options.env = Env::Default();
    options.servable_event_bus = servable_event_bus_.get();
    options.num_load_unload_threads = num_load_unload_threads_;
    options.max_num_load_retries = max_num_load_retries_;
    options.load_retry_interval_micros = 0;

    loader_factory_.reset(new StringLoaderFactory());
    TF_CHECK_OK(CachingManager::Create(std::move(options),
                                       std::move(loader_factory_), &manager_));
  }

  // Creates a manager with a loader-factory that generates errors for all
  // requests. This is to simplify testing for cases related to erroneous
  // handles.
  std::unique_ptr<CachingManager> CreateManagerWithErrorLoaderFactory() {
    num_load_unload_threads_ = GetParam();
    max_num_load_retries_ = 1;

    CachingManager::Options options;
    options.env = Env::Default();
    options.servable_event_bus = servable_event_bus_.get();
    options.num_load_unload_threads = num_load_unload_threads_;
    options.max_num_load_retries = max_num_load_retries_;
    options.load_retry_interval_micros = 0;

    std::unique_ptr<ErrorLoaderFactory> error_loader_factory;
    error_loader_factory.reset(new ErrorLoaderFactory);

    std::unique_ptr<CachingManager> error_manager;
    TF_CHECK_OK(CachingManager::Create(
        std::move(options), std::move(error_loader_factory), &error_manager));
    return error_manager;
  }

  std::shared_ptr<EventBus<ServableState>> servable_event_bus_;
  ServableStateMonitor servable_state_monitor_;
  uint32 num_load_unload_threads_;
  uint32 max_num_load_retries_;
  std::unique_ptr<CachingManager::LoaderFactory> loader_factory_;
  std::unique_ptr<CachingManager> manager_;
};

INSTANTIATE_TEST_CASE_P(WithOrWithoutThreadPool, CachingManagerTest,
                        ::testing::Values(0 /* WithoutThreadPool */, 4));

///////////////////////////////////////////////////////////////////////////////
// Servable handles.

TEST_P(CachingManagerTest, ServableHandleSingleRequest) {
  // Single request for a servable handle.
  ServableHandle<string> handle;
  TF_ASSERT_OK(manager_->GetServableHandle(
      ServableRequest::FromId({kServableName, 0}), &handle));
  EXPECT_EQ("kServableName-0", *handle);
}

TEST_P(CachingManagerTest, ServableHandleMultipleRequests) {
  // Multiple requests in sequence to different servable handles.
  // Scoped to destruct handles at the end of it.
  {
    // Request with servable name and version.
    ServableHandle<string> handle;
    TF_ASSERT_OK(manager_->GetServableHandle(
        ServableRequest::FromId({kServableName, 0}), &handle));
    EXPECT_EQ("kServableName-0", *handle);
  }
  {
    // Request with the same servable name and a higher version.
    ServableHandle<string> handle;
    TF_ASSERT_OK(manager_->GetServableHandle(
        ServableRequest::FromId({kServableName, 1}), &handle));
    EXPECT_EQ("kServableName-1", *handle);
  }
  {
    // Subsequent requests with latest return the loaded servable handle.
    ServableHandle<string> handle;
    TF_ASSERT_OK(manager_->GetServableHandle(
        ServableRequest::Latest({kServableName}), &handle));
    EXPECT_EQ("kServableName-1", *handle);
  }
}

TEST_P(CachingManagerTest, ServableHandleWrongType) {
  // The servable is supposed to be of type string, but we request for a handle
  // of type int. This should cause an invalid argument error.
  ServableHandle<int> handle;
  const Status status = manager_->GetServableHandle(
      ServableRequest::FromId({kServableName, 0}), &handle);
  ASSERT_FALSE(status.ok()) << status;
  EXPECT_EQ(error::INVALID_ARGUMENT, status.code());
}

TEST_P(CachingManagerTest, ServableHandleError) {
  // Create the manager to use the error loader-factory that produces errors.
  std::unique_ptr<CachingManager> error_manager =
      CreateManagerWithErrorLoaderFactory();
  ServableHandle<string> handle;
  const Status status = error_manager->GetServableHandle(
      ServableRequest::FromId({kServableName, 0}), &handle);
  EXPECT_FALSE(status.ok()) << status;
}

///////////////////////////////////////////////////////////////////////////////
// Get available servable handles.

TEST_P(CachingManagerTest, AvailableServableHandlesNoRequests) {
  std::map<ServableId, ServableHandle<string>> handles =
      manager_->GetAvailableServableHandles<string>();
  // Since there are no requests yet, the handles map is empty.
  EXPECT_EQ(0, handles.size());
}

TEST_P(CachingManagerTest, AvailableServableHandlesMultipleRequests) {
  // Multiple requests in sequence to different servable handles.
  // Scoped to destruct handles at the end of it.
  {
    // Request with servable name and version.
    ServableHandle<string> handle;
    TF_ASSERT_OK(manager_->GetServableHandle(
        ServableRequest::FromId({kServableName, 0}), &handle));
  }
  {
    // Request with different version.
    ServableHandle<string> handle;
    TF_ASSERT_OK(manager_->GetServableHandle(
        ServableRequest::FromId({kServableName, 1}), &handle));
  }
  {
    // Request with a different servable name.
    ServableHandle<string> handle;
    TF_ASSERT_OK(manager_->GetServableHandle(
        ServableRequest::FromId({kServableName2, 2}), &handle));
  }
  const std::map<ServableId, ServableHandle<string>> handles =
      manager_->GetAvailableServableHandles<string>();
  std::vector<ServableId> actual_keys;
  for (const auto& it_handle : handles) {
    actual_keys.push_back(it_handle.first);
  }

  const std::vector<ServableId> expected_keys = {
      {kServableName, 0}, {kServableName, 1}, {kServableName2, 2}};
  EXPECT_THAT(actual_keys, UnorderedElementsAreArray(expected_keys));
}

TEST_P(CachingManagerTest, AvailableServableHandlesWrongType) {
  ServableHandle<string> handle;
  TF_ASSERT_OK(manager_->GetServableHandle(
      ServableRequest::FromId({kServableName, 0}), &handle));
  std::map<ServableId, ServableHandle<int>> handles =
      manager_->GetAvailableServableHandles<int>();
  EXPECT_EQ(0, handles.size());
}

TEST_P(CachingManagerTest, AvailableServableHandlesError) {
  // Create the manager to use the error loader-factory that produces errors.
  std::unique_ptr<CachingManager> error_manager =
      CreateManagerWithErrorLoaderFactory();
  ServableHandle<string> handle;
  const Status status = error_manager->GetServableHandle(
      ServableRequest::FromId({kServableName, 0}), &handle);
  ASSERT_FALSE(status.ok()) << status;
  std::map<ServableId, ServableHandle<string>> handles =
      error_manager->GetAvailableServableHandles<string>();
  EXPECT_EQ(0, handles.size());
}

///////////////////////////////////////////////////////////////////////////////
// List available servable ids.

TEST_P(CachingManagerTest, ListAvailableServableIdsMultipleRequests) {
  {
    // Request with servable name and version.
    ServableHandle<string> handle;
    TF_ASSERT_OK(manager_->GetServableHandle(
        ServableRequest::FromId({kServableName, 0}), &handle));
  }
  {
    // Request with a different version.
    ServableHandle<string> handle;
    TF_ASSERT_OK(manager_->GetServableHandle(
        ServableRequest::FromId({kServableName, 1}), &handle));
  }
  {
    // Request with a different servable name.
    ServableHandle<string> handle;
    TF_ASSERT_OK(manager_->GetServableHandle(
        ServableRequest::FromId({kServableName2, 2}), &handle));
  }
  const std::vector<ServableId> expected = {
      {kServableName, 0}, {kServableName, 1}, {kServableName2, 2}};
  EXPECT_THAT(manager_->ListAvailableServableIds(),
              UnorderedElementsAreArray(expected));
}

///////////////////////////////////////////////////////////////////////////////
// Event bus.

MATCHER_P(EqualsServableState, servable_state, servable_state.DebugString()) {
  if (arg == servable_state) {
    return true;
  }
  *result_listener << arg.DebugString();
  return false;
}

TEST_P(CachingManagerTest, EventBusSingleRequest) {
  ServableHandle<string> handle;
  const ServableId id = {kServableName, 0};
  TF_ASSERT_OK(
      manager_->GetServableHandle(ServableRequest::FromId(id), &handle));
  // Check that the state published on the event-bus matches produced by the
  // loader-factory for a successful request.
  const ServableState expected_published_state = {
      id, ServableState::ManagerState::kAvailable, Status::OK()};
  EXPECT_THAT(*servable_state_monitor_.GetState(id),
              EqualsServableState(expected_published_state));
}

TEST_P(CachingManagerTest, EventBusErrorHandle) {
  // Create the manager to use the error loader-factory that produces errors.
  std::unique_ptr<CachingManager> error_manager =
      CreateManagerWithErrorLoaderFactory();
  ServableHandle<string> handle;
  const ServableId id = {kServableName, 0};
  const Status status =
      error_manager->GetServableHandle(ServableRequest::FromId(id), &handle);
  // Check that the state published on the event-bus matches that produced
  // by the loader-factory for an error.
  const ServableState expected_published_state = {
      id, ServableState::ManagerState::kEnd,
      errors::Unknown("error loader-factory")};
  EXPECT_THAT(*servable_state_monitor_.GetState(id),
              EqualsServableState(expected_published_state));
}

///////////////////////////////////////////////////////////////////////////////
// Concurrent requests.
// TODO(b/25449742): Update with test for the same ServableId once the caching
// manager has support for multithreading.

TEST_P(CachingManagerTest, ServableHandleConcurrentRequestsDifferentIds) {
  // Track the status of each request.
  mutex status_mu;
  std::vector<Status> statuses(4);
  {
    ThreadPoolExecutor request_executor(Env::Default(), "GetHandles",
                                        kNumThreads);
    for (int i = 0; i < 4; i++) {
      request_executor.Schedule([this, i, &statuses, &status_mu]() {
        ServableHandle<string> handle;
        const Status status =
            manager_->GetServableHandle({kServableName, i}, &handle);
        mutex_lock l(status_mu);
        statuses[i] = status;
      });
    }
  }
  // Check that all requests returned with an ok status.
  for (int i = 0; i < 4; i++) {
    mutex_lock l(status_mu);
    EXPECT_EQ(Status::OK(), statuses[i]);
  }
  // Check that the available servable handles now includes all requested
  // servables.
  const std::map<ServableId, ServableHandle<string>> handles =
      manager_->GetAvailableServableHandles<string>();
  std::vector<ServableId> actual_keys;
  for (const auto& it_handle : handles) {
    actual_keys.push_back(it_handle.first);
  }

  const std::vector<ServableId> expected_keys = {{kServableName, 0},
                                                 {kServableName, 1},
                                                 {kServableName, 2},
                                                 {kServableName, 3}};
  EXPECT_THAT(actual_keys, UnorderedElementsAreArray(expected_keys));
}

///////////////////////////////////////////////////////////////////////////////

}  // namespace
}  // namespace serving
}  // namespace tensorflow
