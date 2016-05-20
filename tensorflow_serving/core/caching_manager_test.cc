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
#include "tensorflow_serving/core/test_util/manager_test_util.h"
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
  explicit StringLoaderFactory(const int64 starting_version)
      : latest_version_(starting_version) {}

  ~StringLoaderFactory() override = default;

  Status CreateLoader(
      const ServableId& id,
      std::unique_ptr<ServableData<std::unique_ptr<Loader>>>* loaded_data) {
    auto servable_creator = [&](std::unique_ptr<string>* servable) {
      servable->reset(new string);
      **servable = strings::StrCat(id.name, "-", id.version);
      return Status::OK();
    };
    std::unique_ptr<Loader> loader;
    loader.reset(new SimpleLoader<string>(
        servable_creator, SimpleLoader<string>::EstimateNoResources()));
    loaded_data->reset(
        new ServableData<std::unique_ptr<Loader>>(id, std::move(loader)));
    // Update state to indicate a new loader was created.
    mutex_lock l(mu_);
    num_loaders_dispensed_++;
    return Status::OK();
  }

  // Returns the latest version corresponding to the servable name.
  int64 GetLatestVersion(const string& request_name) const {
    // Increment the current latest version until a maximum of 42.
    mutex_lock l(mu_);
    return latest_version_;
  }

  // Update the latest available version.
  void set_latest_version(int64 version) {
    mutex_lock l(mu_);
    latest_version_ = version;
  }

  // Returns the number of loaders created by the loader-factory.
  int64 num_loaders_dispensed() const {
    mutex_lock l(mu_);
    return num_loaders_dispensed_;
  }

 private:
  // Used to protect updates to the latest_version_.
  mutable mutex mu_;

  // The current latest version.
  int64 latest_version_ GUARDED_BY(mu_) = 0;

  // Tracks the number of loaders dispensed by the loader-factory.
  int64 num_loaders_dispensed_ GUARDED_BY(mu_) = 0;

  TF_DISALLOW_COPY_AND_ASSIGN(StringLoaderFactory);
};

// A simple loader-factory that always returns a loader with an error for every
// request.
class ErrorLoaderFactory : public CachingManager::LoaderFactory {
 public:
  ErrorLoaderFactory() = default;
  ~ErrorLoaderFactory() override = default;

  Status CreateLoader(
      const ServableId& id,
      std::unique_ptr<ServableData<std::unique_ptr<Loader>>>* loaded_data) {
    auto servable_creator = [&](std::unique_ptr<string>* servable) {
      return errors::Unknown("error loader-factory");
    };
    std::unique_ptr<Loader> loader;
    loader.reset(new SimpleLoader<string>(
        servable_creator, SimpleLoader<string>::EstimateNoResources()));
    loaded_data->reset(
        new ServableData<std::unique_ptr<Loader>>(id, std::move(loader)));
    return Status::OK();
  }

  int64 GetLatestVersion(const string& request_name) const {
    // A simple "latest" interpretation that always returns version 42.
    return 42;
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
    CachingManager::Options options;
    options.env = Env::Default();
    options.servable_event_bus = servable_event_bus_.get();
    options.num_load_unload_threads = GetParam();
    options.max_num_load_retries = 1;
    options.load_retry_interval_micros = 0;

    std::unique_ptr<StringLoaderFactory> string_loader_factory;
    string_loader_factory.reset(new StringLoaderFactory(0));
    string_loader_factory_ = string_loader_factory.get();

    TF_CHECK_OK(CachingManager::Create(
        std::move(options), std::move(string_loader_factory), &manager_));
  }

  // Creates a manager with a loader-factory that generates errors for all
  // requests. This is to simplify testing for cases related to erroneous
  // handles.
  std::unique_ptr<CachingManager> CreateManagerWithErrorLoaderFactory() {
    CachingManager::Options options;
    options.env = Env::Default();
    options.servable_event_bus = servable_event_bus_.get();
    options.num_load_unload_threads = GetParam();
    options.max_num_load_retries = 1;
    options.load_retry_interval_micros = 0;

    std::unique_ptr<ErrorLoaderFactory> error_loader_factory;
    error_loader_factory.reset(new ErrorLoaderFactory);

    std::unique_ptr<CachingManager> error_manager;
    TF_CHECK_OK(CachingManager::Create(
        std::move(options), std::move(error_loader_factory), &error_manager));
    return error_manager;
  }

  // Helper function to return the size of the load-mutex map from the
  // caching-manager.
  int64 GetLoadMutexMapSize() {
    return test_util::CachingManagerTestAccess(manager_.get())
        .GetLoadMutexMapSize();
  }

  std::shared_ptr<EventBus<ServableState>> servable_event_bus_;
  ServableStateMonitor servable_state_monitor_;
  std::unique_ptr<CachingManager> manager_;
  StringLoaderFactory* string_loader_factory_;
};

INSTANTIATE_TEST_CASE_P(WithOrWithoutThreadPool, CachingManagerTest,
                        ::testing::Values(0 /* WithoutThreadPool */, 4));

///////////////////////////////////////////////////////////////////////////////
// Servable handles.

TEST_P(CachingManagerTest, ServableHandleSingleRequest) {
  // Single request for a servable handle.
  const ServableId id = {kServableName, 30};
  ServableHandle<string> handle;
  TF_ASSERT_OK(
      manager_->GetServableHandle(ServableRequest::FromId(id), &handle));
  EXPECT_EQ("kServableName-30", *handle);
  EXPECT_EQ(id, handle.id());
}

TEST_P(CachingManagerTest, ServableHandleMultipleRequests) {
  // Multiple requests in sequence to different servable handles.
  // Scoped to destruct handles at the end of it.
  {
    // Request with servable name and version.
    const ServableId id = {kServableName, 30};
    ServableHandle<string> handle;
    TF_ASSERT_OK(
        manager_->GetServableHandle(ServableRequest::FromId(id), &handle));
    EXPECT_EQ("kServableName-30", *handle);
    EXPECT_EQ(id, handle.id());
  }
  {
    // Request with the same servable name and a higher version.
    const ServableId id = {kServableName, 31};
    ServableHandle<string> handle;
    TF_ASSERT_OK(
        manager_->GetServableHandle(ServableRequest::FromId(id), &handle));
    EXPECT_EQ("kServableName-31", *handle);
    EXPECT_EQ(id, handle.id());
  }
}

// Tests functionality when the version corresponding to the "latest" needs to
// be newly managed and loaded by the manager.
TEST_P(CachingManagerTest, ServableHandleSingleRequestLatest) {
  string_loader_factory_->set_latest_version(30);
  ServableHandle<string> handle;
  TF_ASSERT_OK(manager_->GetServableHandle(
      ServableRequest::Latest({kServableName}), &handle));
  EXPECT_EQ("kServableName-30", *handle);
  const ServableId id = {kServableName, 30};
  EXPECT_EQ(id, handle.id());
}

// Tests functionality when the version corresponding to the "latest" is
// already managed and loaded by the caching-manager.
TEST_P(CachingManagerTest, ServableHandleMultipleRequestsLatest) {
  const ServableId id = {kServableName, 42};
  {
    // Make an explicit request for version 42.
    ServableHandle<string> handle;
    TF_ASSERT_OK(
        manager_->GetServableHandle(ServableRequest::FromId(id), &handle));
    EXPECT_EQ("kServableName-42", *handle);
    EXPECT_EQ(id, handle.id());
    // We expect a new loader to be created for this request.
    EXPECT_EQ(1, string_loader_factory_->num_loaders_dispensed());
    // Update the latest available version.
    string_loader_factory_->set_latest_version(42);
  }
  {
    // Now request for the latest. The returned handle should have an id
    // corresponding to version 42.
    ServableHandle<string> handle;
    TF_ASSERT_OK(manager_->GetServableHandle(
        ServableRequest::Latest({kServableName}), &handle));
    EXPECT_EQ("kServableName-42", *handle);
    EXPECT_EQ(id, handle.id());
    // We do not expect a new loader to be created for this request, since it is
    // identical to the previous request.
    EXPECT_EQ(1, string_loader_factory_->num_loaders_dispensed());
  }
}

TEST_P(CachingManagerTest, ServableHandleWrongType) {
  // The servable is supposed to be of type string, but we request for a handle
  // of type int. This should cause an invalid argument error.
  ServableHandle<int> handle;
  const Status status = manager_->GetServableHandle(
      ServableRequest::FromId({kServableName, 30}), &handle);
  ASSERT_FALSE(status.ok()) << status;
  EXPECT_EQ(error::INVALID_ARGUMENT, status.code());
}

TEST_P(CachingManagerTest, ServableHandleError) {
  // Create the manager to use the error loader-factory that produces errors.
  std::unique_ptr<CachingManager> error_manager =
      CreateManagerWithErrorLoaderFactory();
  ServableHandle<string> handle;
  const Status status = error_manager->GetServableHandle(
      ServableRequest::FromId({kServableName, 30}), &handle);
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
        ServableRequest::FromId({kServableName, 30}), &handle));
  }
  {
    // Request with different version.
    ServableHandle<string> handle;
    TF_ASSERT_OK(manager_->GetServableHandle(
        ServableRequest::FromId({kServableName, 31}), &handle));
  }
  {
    // Request with a different servable name.
    ServableHandle<string> handle;
    TF_ASSERT_OK(manager_->GetServableHandle(
        ServableRequest::FromId({kServableName2, 32}), &handle));
  }
  const std::map<ServableId, ServableHandle<string>> handles =
      manager_->GetAvailableServableHandles<string>();
  std::vector<ServableId> actual_keys;
  for (const auto& it_handle : handles) {
    actual_keys.push_back(it_handle.first);
  }

  const std::vector<ServableId> expected_keys = {
      {kServableName, 30}, {kServableName, 31}, {kServableName2, 32}};
  EXPECT_THAT(actual_keys, UnorderedElementsAreArray(expected_keys));
}

TEST_P(CachingManagerTest, AvailableServableHandlesWrongType) {
  ServableHandle<string> handle;
  TF_ASSERT_OK(manager_->GetServableHandle(
      ServableRequest::FromId({kServableName, 30}), &handle));
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
      ServableRequest::FromId({kServableName, 30}), &handle);
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
        ServableRequest::FromId({kServableName, 30}), &handle));
  }
  {
    // Request with a different version.
    ServableHandle<string> handle;
    TF_ASSERT_OK(manager_->GetServableHandle(
        ServableRequest::FromId({kServableName, 31}), &handle));
  }
  {
    // Request with a different servable name.
    ServableHandle<string> handle;
    TF_ASSERT_OK(manager_->GetServableHandle(
        ServableRequest::FromId({kServableName2, 32}), &handle));
  }
  const std::vector<ServableId> expected = {
      {kServableName, 30}, {kServableName, 31}, {kServableName2, 32}};
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
  const ServableId id = {kServableName, 30};
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
  const ServableId id = {kServableName, 30};
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

TEST_P(CachingManagerTest, ConcurrentDisjointRequests) {
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
            manager_->GetServableHandle({kServableName, i + 30}, &handle);
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

  const std::vector<ServableId> expected_keys = {{kServableName, 30},
                                                 {kServableName, 31},
                                                 {kServableName, 32},
                                                 {kServableName, 33}};
  EXPECT_THAT(actual_keys, UnorderedElementsAreArray(expected_keys));
  // Since the map entries in load_mutex_map_ are garbage-collected, we expect
  // no remaining entries in the map.
  EXPECT_EQ(0, GetLoadMutexMapSize());
}

TEST_P(CachingManagerTest, ConcurrentIntersectingRequests) {
  mutex status_mu;
  std::vector<Status> statuses(8);
  {
    ThreadPoolExecutor request_executor(Env::Default(), "GetHandles",
                                        kNumThreads);
    for (int i = 0; i < 8; i++) {
      // Use two different versions to send concurrent requests.
      const int version = i % 2 + 30;
      const ServableId id = {kServableName, version};
      request_executor.Schedule([this, i, id, &statuses, &status_mu]() {
        ServableHandle<string> handle;
        const Status status =
            manager_->GetServableHandle(ServableRequest::FromId(id), &handle);
        mutex_lock l(status_mu);
        statuses[i] = status;
      });
    }
  }
  // Check that all requests returned with an ok status.
  for (int i = 0; i < 8; i++) {
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
  const std::vector<ServableId> expected_keys = {{kServableName, 30},
                                                 {kServableName, 31}};
  EXPECT_THAT(actual_keys, UnorderedElementsAreArray(expected_keys));
  // Since the map entries in load_mutex_map_ are garbage-collected, we expect
  // no remaining entries in the map.
  EXPECT_EQ(0, GetLoadMutexMapSize());
}

///////////////////////////////////////////////////////////////////////////////

}  // namespace
}  // namespace serving
}  // namespace tensorflow
