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

#include "tensorflow_serving/core/aspired_versions_manager.h"

#include <algorithm>
#include <functional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow_serving/core/availability_preserving_policy.h"
#include "tensorflow_serving/core/servable_state_monitor.h"
#include "tensorflow_serving/core/test_util/availability_test_util.h"
#include "tensorflow_serving/core/test_util/fake_loader.h"
#include "tensorflow_serving/core/test_util/manager_test_util.h"
#include "tensorflow_serving/core/test_util/mock_loader.h"
#include "tensorflow_serving/util/any_ptr.h"
#include "tensorflow_serving/util/event_bus.h"

namespace tensorflow {
namespace serving {
namespace {

using ::testing::_;
using ::testing::DoAll;
using ::testing::Invoke;
using ::testing::InvokeWithoutArgs;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::UnorderedElementsAre;
using ::testing::UnorderedElementsAreArray;
using test_util::FakeLoader;
using test_util::WaitUntilServableManagerStateIsOneOf;

constexpr char kServableName[] = "kServableName";
constexpr char kServableName2[] = "kServableName2";
constexpr int kNumVersionsPerServable = 2;
constexpr int kNumTotalVersions = 4;

// Creates an aspired-versions entry with 'id' and a FakeLoader whose servable
// is id.version.
ServableData<std::unique_ptr<Loader>> CreateAspiredVersion(
    const ServableId& id) {
  std::unique_ptr<Loader> loader(new FakeLoader(id.version));
  return CreateServableData(id, std::move(loader));
}

// We parameterize this test with the number of load & unload threads. (Zero
// means use an in-line executor instead of a thread pool.)
struct ThreadPoolSizes {
  uint64 num_load_threads;
  uint64 num_unload_threads;
};
class AspiredVersionsManagerTest
    : public ::testing::TestWithParam<ThreadPoolSizes> {
 protected:
  AspiredVersionsManagerTest()
      : servable_event_bus_(EventBus<ServableState>::CreateEventBus()),
        servable_state_monitor_(servable_event_bus_.get()),
        thread_pool_sizes_(GetParam()) {
    AspiredVersionsManager::Options manager_options;
    manager_options.num_load_threads = thread_pool_sizes_.num_load_threads;
    manager_options.num_unload_threads = thread_pool_sizes_.num_unload_threads;
    // The state manager thread won't be run automatically.
    manager_options.manage_state_interval_micros = -1;
    manager_options.env = Env::Default();
    manager_options.aspired_version_policy.reset(
        new AvailabilityPreservingPolicy());
    manager_options.servable_event_bus = servable_event_bus_.get();
    max_num_load_retries_ = 1;
    manager_options.max_num_load_retries = max_num_load_retries_;
    manager_options.load_retry_interval_micros = 0;
    TF_CHECK_OK(
        AspiredVersionsManager::Create(std::move(manager_options), &manager_));
  }

  // Creates an aspired-versions entry with 'id' and an error (and no loader).
  ServableData<std::unique_ptr<Loader>> CreateErroneousAspiredVersion(
      const ServableId& id) {
    return ServableData<std::unique_ptr<Loader>>(id, errors::Unknown("error"));
  }

  void SetUp() override {
    // We setUp the manager_ with two different servable streams, each with two
    // aspired versions 0 and 1.
    std::set<ServableId> servables;
    std::vector<ServableData<std::unique_ptr<Loader>>> aspired_versions;
    for (int i = 0; i < kNumVersionsPerServable; ++i) {
      const ServableId id = {kServableName, i};
      aspired_versions.push_back(CreateAspiredVersion(id));
      servables.insert(id);
    }
    manager_->GetAspiredVersionsCallback()(kServableName,
                                           std::move(aspired_versions));
    HandlePendingAspiredVersionsRequests();

    std::vector<ServableData<std::unique_ptr<Loader>>> aspired_versions2;
    for (int i = 0; i < kNumVersionsPerServable; ++i) {
      const ServableId id = {kServableName2, i};
      aspired_versions2.push_back(CreateAspiredVersion(id));
      servables.insert(id);
    }
    manager_->GetAspiredVersionsCallback()(kServableName2,
                                           std::move(aspired_versions2));
    HandlePendingAspiredVersionsRequests();

    for (int i = 0; i < kNumTotalVersions; ++i) {
      // Each time the state manager thread is run, we should load a servable
      // version.
      InvokePolicyAndExecuteAction();
    }
    for (const ServableId& servable : servables) {
      WaitUntilServableManagerStateIsOneOf(
          servable_state_monitor_, servable,
          {ServableState::ManagerState::kAvailable});
    }
  }

  void FlushServables() {
    test_util::AspiredVersionsManagerTestAccess(manager_.get())
        .FlushServables();
  }

  void HandlePendingAspiredVersionsRequests() {
    test_util::AspiredVersionsManagerTestAccess(manager_.get())
        .HandlePendingAspiredVersionsRequests();
  }

  void InvokePolicyAndExecuteAction() {
    test_util::AspiredVersionsManagerTestAccess(manager_.get())
        .InvokePolicyAndExecuteAction();
  }

  std::shared_ptr<EventBus<ServableState>> servable_event_bus_;
  ServableStateMonitor servable_state_monitor_;
  ThreadPoolSizes thread_pool_sizes_;
  uint32 max_num_load_retries_;
  std::unique_ptr<AspiredVersionsManager> manager_;
};

INSTANTIATE_TEST_CASE_P(
    WithOrWithoutThreadPools, AspiredVersionsManagerTest,
    ::testing::Values(
        ThreadPoolSizes{0, 0} /* without load or unload threadpools */,
        ThreadPoolSizes{2, 0} /* with just a load threadpool */,
        ThreadPoolSizes{0, 2} /* with just an unload threadpool */,
        ThreadPoolSizes{4, 4} /* with load and unload threadpools */));

TEST_P(AspiredVersionsManagerTest, ServableHandleNotFoundMissingLoaderName) {
  ServableHandle<int64> handle;
  const Status status = manager_->GetServableHandle(
      ServableRequest::Latest(strings::StrCat(kServableName, "missing")),
      &handle);
  ASSERT_FALSE(status.ok()) << status;
  EXPECT_EQ(error::NOT_FOUND, status.code());
}

TEST_P(AspiredVersionsManagerTest, ServableHandleNotFoundMissingVersion) {
  // This version is missing.
  const int64 missing_version = 100;
  ServableHandle<int64> handle;
  const Status status = manager_->GetServableHandle(
      ServableRequest::Specific(kServableName, missing_version), &handle);
  ASSERT_FALSE(status.ok()) << status;
  EXPECT_EQ(error::NOT_FOUND, status.code());
}

TEST_P(AspiredVersionsManagerTest, ServableHandleInvalidArgument) {
  // The servable is supposed to be an int type and we ask for a float type,
  // thus causing an invalid argument error.
  ServableHandle<float> handle;
  const Status status = manager_->GetServableHandle(
      ServableRequest::Latest(kServableName), &handle);
  ASSERT_FALSE(status.ok()) << status;
  EXPECT_EQ(error::INVALID_ARGUMENT, status.code());
}

TEST_P(AspiredVersionsManagerTest, ServableHandleLatest) {
  std::vector<ServableData<std::unique_ptr<Loader>>> aspired_versions;
  const ServableId id = {kServableName, kNumVersionsPerServable + 1};
  aspired_versions.push_back(CreateAspiredVersion(id));
  manager_->GetAspiredVersionsCallback()(kServableName,
                                         std::move(aspired_versions));
  HandlePendingAspiredVersionsRequests();
  // Unload version 0 and load the new aspired version. Version 1 may or may not
  // be unloaded (depending on whether load/unload thread pools are used).
  for (int i = 0; i < kNumVersionsPerServable + 1; ++i) {
    InvokePolicyAndExecuteAction();
  }
  WaitUntilServableManagerStateIsOneOf(
      servable_state_monitor_, id, {ServableState::ManagerState::kAvailable});

  ServableHandle<int64> handle;
  const Status status = manager_->GetServableHandle(
      ServableRequest::Latest(kServableName), &handle);
  TF_ASSERT_OK(status);
  EXPECT_EQ(kNumVersionsPerServable + 1, *handle);
}

// Test the case where the latest version of a servable available is 0.
TEST_P(AspiredVersionsManagerTest, ServableHandleLatestVersionIsZero) {
  const char kServableName3[] = "kServableName3";

  std::vector<ServableData<std::unique_ptr<Loader>>> aspired_versions;
  const ServableId id = {kServableName3, 0};
  aspired_versions.push_back(CreateAspiredVersion(id));
  manager_->GetAspiredVersionsCallback()(kServableName3,
                                         std::move(aspired_versions));
  HandlePendingAspiredVersionsRequests();

  InvokePolicyAndExecuteAction();
  WaitUntilServableManagerStateIsOneOf(
      servable_state_monitor_, id, {ServableState::ManagerState::kAvailable});

  ServableHandle<int64> handle;
  const Status status = manager_->GetServableHandle(
      ServableRequest::Latest(kServableName3), &handle);
  TF_ASSERT_OK(status);
  EXPECT_EQ(0, *handle);
  EXPECT_EQ(id, handle.id());
}

TEST_P(AspiredVersionsManagerTest, ServableHandleSpecificVersion) {
  ServableHandle<int64> handle;
  const ServableId id = {kServableName2, 0};
  const Status status =
      manager_->GetServableHandle(ServableRequest::FromId(id), &handle);
  TF_ASSERT_OK(status);
  EXPECT_EQ(0, *handle);
  EXPECT_EQ(id, handle.id());
}

TEST_P(AspiredVersionsManagerTest, ListAvailableServableIds) {
  const std::vector<ServableId> expected_before = {{kServableName, 0},
                                                   {kServableName, 1},
                                                   {kServableName2, 0},
                                                   {kServableName2, 1}};
  EXPECT_THAT(manager_->ListAvailableServableIds(),
              UnorderedElementsAreArray(expected_before));

  // Set stream kServableName to have servables 7.
  // This causes 0 & 1 to be unloaded and 7 to be loaded, but 7 errors on load,
  // so never moves to a loaded state.
  std::vector<ServableData<std::unique_ptr<Loader>>> aspired_versions;
  const ServableId id = {kServableName, 7};
  std::unique_ptr<Loader> loader(
      new FakeLoader(7, errors::Internal("An error.")));
  aspired_versions.push_back({id, std::move(loader)});
  manager_->GetAspiredVersionsCallback()(kServableName,
                                         std::move(aspired_versions));
  HandlePendingAspiredVersionsRequests();
  for (int i = 0; i < kNumVersionsPerServable + 1; ++i) {
    InvokePolicyAndExecuteAction();
  }
  WaitUntilServableManagerStateIsOneOf(servable_state_monitor_, id,
                                       {ServableState::ManagerState::kEnd});

  manager_->GetAspiredVersionsCallback()(kServableName, {});
  HandlePendingAspiredVersionsRequests();
  for (int i = 0; i < kNumVersionsPerServable + 1; ++i) {
    InvokePolicyAndExecuteAction();
  }
  WaitUntilServableManagerStateIsOneOf(servable_state_monitor_,
                                       {kServableName, 0},
                                       {ServableState::ManagerState::kEnd});
  WaitUntilServableManagerStateIsOneOf(servable_state_monitor_,
                                       {kServableName, 1},
                                       {ServableState::ManagerState::kEnd});

  const std::vector<ServableId> expected_after = {{kServableName2, 0},
                                                  {kServableName2, 1}};
  EXPECT_THAT(manager_->ListAvailableServableIds(),
              UnorderedElementsAreArray(expected_after));
}

TEST_P(AspiredVersionsManagerTest, GetAvailableServableHandles) {
  // Scoped to destruct handles at the end of it.
  {
    const std::map<ServableId, ServableHandle<int64>> handles_before =
        manager_->GetAvailableServableHandles<int64>();
    ASSERT_EQ(kNumVersionsPerServable * 2, handles_before.size());

    const std::vector<ServableId> expected_ids_before = {{kServableName, 0},
                                                         {kServableName, 1},
                                                         {kServableName2, 0},
                                                         {kServableName2, 1}};
    for (const ServableId& expected_id : expected_ids_before) {
      const auto found_it = handles_before.find(expected_id);
      ASSERT_TRUE(found_it != handles_before.end());
      EXPECT_EQ(expected_id.version, *found_it->second);
    }
  }

  // Set stream kServableName to have servables 7.
  // This causes 0 & 1 to be unloaded and 7 to be loaded, but 7 errors on load,
  // so never moves to a loaded state.
  std::vector<ServableData<std::unique_ptr<Loader>>> aspired_versions;
  const ServableId id = {kServableName, 7};
  std::unique_ptr<Loader> loader(
      new FakeLoader(7, errors::Internal("An error.")));
  aspired_versions.push_back({id, std::move(loader)});
  manager_->GetAspiredVersionsCallback()(kServableName,
                                         std::move(aspired_versions));
  HandlePendingAspiredVersionsRequests();
  for (int i = 0; i < kNumVersionsPerServable + 1; ++i) {
    InvokePolicyAndExecuteAction();
  }
  WaitUntilServableManagerStateIsOneOf(servable_state_monitor_, id,
                                       {ServableState::ManagerState::kEnd});

  manager_->GetAspiredVersionsCallback()(kServableName, {});
  HandlePendingAspiredVersionsRequests();
  for (int i = 0; i < kNumVersionsPerServable + 1; ++i) {
    InvokePolicyAndExecuteAction();
  }
  WaitUntilServableManagerStateIsOneOf(servable_state_monitor_,
                                       {kServableName, 0},
                                       {ServableState::ManagerState::kEnd});
  WaitUntilServableManagerStateIsOneOf(servable_state_monitor_,
                                       {kServableName, 1},
                                       {ServableState::ManagerState::kEnd});
  {
    const std::map<ServableId, ServableHandle<int64>> handles_after =
        manager_->GetAvailableServableHandles<int64>();
    ASSERT_EQ(kNumVersionsPerServable, handles_after.size());

    const std::vector<ServableId> expected_ids_after = {{kServableName2, 0},
                                                        {kServableName2, 1}};
    for (const ServableId& expected_id : expected_ids_after) {
      const auto found_it = handles_after.find(expected_id);
      ASSERT_TRUE(found_it != handles_after.end());
      EXPECT_EQ(expected_id.version, *found_it->second);
    }
  }
}

TEST_P(AspiredVersionsManagerTest, GetAvailableServableHandlesWrongType) {
  const std::map<ServableId, ServableHandle<int>> wrong_type_handles =
      manager_->GetAvailableServableHandles<int>();
  EXPECT_EQ(0, wrong_type_handles.size());
}

TEST_P(AspiredVersionsManagerTest, AspiredRemovedFull) {
  // Scoped so that the handle is destructed at the end, and the harness is
  // destructed when we run the manager looping thread.
  {
    ServableHandle<int64> handle;
    const Status status = manager_->GetServableHandle(
        ServableRequest::Latest(kServableName), &handle);
    TF_ASSERT_OK(status);
    EXPECT_EQ(1, *handle);
  }

  manager_->GetAspiredVersionsCallback()(kServableName, {});
  HandlePendingAspiredVersionsRequests();

  const int num_fake_loaders_before = FakeLoader::num_fake_loaders();
  for (int i = 0; i < kNumVersionsPerServable; ++i) {
    InvokePolicyAndExecuteAction();
  }
  WaitUntilServableManagerStateIsOneOf(servable_state_monitor_,
                                       {kServableName, 0},
                                       {ServableState::ManagerState::kEnd});
  WaitUntilServableManagerStateIsOneOf(servable_state_monitor_,
                                       {kServableName, 1},
                                       {ServableState::ManagerState::kEnd});
  FlushServables();
  const int num_fake_loaders_after = FakeLoader::num_fake_loaders();
  EXPECT_EQ(kNumVersionsPerServable,
            num_fake_loaders_before - num_fake_loaders_after);

  ServableHandle<int64> missing_handle;
  const Status missing_status = manager_->GetServableHandle(
      ServableRequest::Latest(kServableName), &missing_handle);
  ASSERT_FALSE(missing_status.ok());
  EXPECT_EQ(error::NOT_FOUND, missing_status.code());
}

TEST_P(AspiredVersionsManagerTest, AspiredRemovedPartial) {
  std::vector<ServableData<std::unique_ptr<Loader>>> aspired_versions;
  aspired_versions.push_back(CreateAspiredVersion({kServableName, 0}));
  manager_->GetAspiredVersionsCallback()(kServableName,
                                         std::move(aspired_versions));
  HandlePendingAspiredVersionsRequests();

  InvokePolicyAndExecuteAction();
  WaitUntilServableManagerStateIsOneOf(servable_state_monitor_,
                                       {kServableName, 1},
                                       {ServableState::ManagerState::kEnd});

  // Version 0 should remain available in the manager.
  ServableHandle<int64> v0_handle;
  const Status v0_status = manager_->GetServableHandle(
      ServableRequest::Specific(kServableName, 0), &v0_handle);
  TF_ASSERT_OK(v0_status);
  EXPECT_EQ(0, *v0_handle);

  // Version 1 should no longer be available.
  ServableHandle<int64> v1_handle;
  const Status v1_status = manager_->GetServableHandle(
      ServableRequest::Specific(kServableName, 1), &v1_handle);
  ASSERT_FALSE(v1_status.ok());
  EXPECT_EQ(error::NOT_FOUND, v1_status.code());
}

TEST_P(AspiredVersionsManagerTest, RevertToSmallerVersionNumber) {
  // Initially, versions 0 and 1 of kServableName are loaded.
  std::set<int64> initial_versions;
  for (const ServableId& id : manager_->ListAvailableServableIds()) {
    if (id.name == kServableName) {
      initial_versions.insert(id.version);
    }
  }
  ASSERT_THAT(initial_versions, UnorderedElementsAre(0, 1));

  // Unload version 0, s.t. only version 1 is loaded.
  std::vector<ServableData<std::unique_ptr<Loader>>> initial_aspired_versions;
  initial_aspired_versions.push_back(CreateAspiredVersion({kServableName, 1}));
  manager_->GetAspiredVersionsCallback()(kServableName,
                                         std::move(initial_aspired_versions));
  HandlePendingAspiredVersionsRequests();
  InvokePolicyAndExecuteAction();
  WaitUntilServableManagerStateIsOneOf(servable_state_monitor_,
                                       {kServableName, 0},
                                       {ServableState::ManagerState::kEnd});
  FlushServables();

  // Now, switch to version 0 (dropping version 1).
  std::vector<ServableData<std::unique_ptr<Loader>>> new_aspired_versions;
  new_aspired_versions.push_back(CreateAspiredVersion({kServableName, 0}));
  manager_->GetAspiredVersionsCallback()(kServableName,
                                         std::move(new_aspired_versions));
  HandlePendingAspiredVersionsRequests();
  Notification done_transitioning;
  std::unique_ptr<Thread> transition_servables(
      Env::Default()->StartThread({}, "TransitionServables", [&]() {
        while (!done_transitioning.HasBeenNotified()) {
          InvokePolicyAndExecuteAction();
          Env::Default()->SleepForMicroseconds(1000 /* 1 ms */);
        }
      }));
  WaitUntilServableManagerStateIsOneOf(
      servable_state_monitor_, {kServableName, 0},
      {ServableState::ManagerState::kAvailable});
  WaitUntilServableManagerStateIsOneOf(servable_state_monitor_,
                                       {kServableName, 1},
                                       {ServableState::ManagerState::kEnd});
  done_transitioning.Notify();

  // Version 0 should be available.
  ServableHandle<int64> v0_handle;
  const Status v0_status = manager_->GetServableHandle(
      ServableRequest::Specific(kServableName, 0), &v0_handle);
  TF_ASSERT_OK(v0_status);
  EXPECT_EQ(0, *v0_handle);

  // Version 1 should not be available.
  ServableHandle<int64> v1_handle;
  const Status v1_status = manager_->GetServableHandle(
      ServableRequest::Specific(kServableName, 1), &v1_handle);
  ASSERT_FALSE(v1_status.ok());
  EXPECT_EQ(error::NOT_FOUND, v1_status.code());
}

TEST_P(AspiredVersionsManagerTest, AspiredAndManageStateLoad) {
  const ServableId id = {kServableName, 2};
  ServableHandle<int64> not_found_handle;
  const Status not_found_status = manager_->GetServableHandle(
      ServableRequest::FromId(id), &not_found_handle);
  ASSERT_FALSE(not_found_status.ok()) << not_found_status;
  EXPECT_EQ(error::NOT_FOUND, not_found_status.code());

  std::vector<ServableData<std::unique_ptr<Loader>>> aspired_versions;
  aspired_versions.push_back(CreateAspiredVersion(id));
  manager_->GetAspiredVersionsCallback()(kServableName,
                                         std::move(aspired_versions));
  HandlePendingAspiredVersionsRequests();

  ServableHandle<int64> not_ready_handle;
  const Status not_ready_status = manager_->GetServableHandle(
      ServableRequest::FromId(id), &not_ready_handle);
  ASSERT_FALSE(not_ready_status.ok()) << not_ready_status;
  EXPECT_EQ(error::NOT_FOUND, not_ready_status.code());

  // Unload version 0 and load the new aspired version. Version 1 may or may not
  // be unloaded (depending on whether load/unload thread pools are used).
  for (int i = 0; i < kNumVersionsPerServable + 1; ++i) {
    InvokePolicyAndExecuteAction();
  }
  WaitUntilServableManagerStateIsOneOf(
      servable_state_monitor_, id, {ServableState::ManagerState::kAvailable});

  ServableHandle<int64> handle;
  const Status status =
      manager_->GetServableHandle(ServableRequest::FromId(id), &handle);
  TF_ASSERT_OK(status);
  EXPECT_EQ(2, *handle);
}

TEST_P(AspiredVersionsManagerTest, AspiredAndManageStateUnload) {
  {
    ServableHandle<int64> handle;
    const Status status = manager_->GetServableHandle(
        ServableRequest::Specific(kServableName, 0), &handle);
    TF_ASSERT_OK(status);
    EXPECT_EQ(0, *handle);
  }

  manager_->GetAspiredVersionsCallback()(kServableName, {});
  HandlePendingAspiredVersionsRequests();

  for (int i = 0; i < kNumVersionsPerServable; ++i) {
    InvokePolicyAndExecuteAction();
  }
  WaitUntilServableManagerStateIsOneOf(servable_state_monitor_,
                                       {kServableName, 0},
                                       {ServableState::ManagerState::kEnd});
  WaitUntilServableManagerStateIsOneOf(servable_state_monitor_,
                                       {kServableName, 1},
                                       {ServableState::ManagerState::kEnd});

  ServableHandle<int64> not_found_handle;
  const Status not_found_status = manager_->GetServableHandle(
      ServableRequest::Specific(kServableName, 0), &not_found_handle);
  ASSERT_FALSE(not_found_status.ok()) << not_found_status;
  EXPECT_EQ(error::NOT_FOUND, not_found_status.code());
}

// The manager prefers unloading over loading when deciding between different
// servable actions. This behaviour is tested here.
TEST_P(AspiredVersionsManagerTest, ManagerPrefersUnloadOverLoad) {
  ServableHandle<int64> not_found_2_handle;
  Status not_found_2_status = manager_->GetServableHandle(
      ServableRequest::Specific(kServableName2, 2), &not_found_2_handle);
  ASSERT_FALSE(not_found_2_status.ok()) << not_found_2_status;
  EXPECT_EQ(error::NOT_FOUND, not_found_2_status.code());

  {
    ServableHandle<int64> found_0_handle;
    const Status found_0_status = manager_->GetServableHandle(
        ServableRequest::Specific(kServableName, 0), &found_0_handle);
    TF_ASSERT_OK(found_0_status);
    EXPECT_EQ(0, *found_0_handle);
  }

  // We want to unload version 0 of the first servable stream and load version 2
  // of the second stream.
  struct {
    StringPiece name;
    int start;
    int end;
  } servable_aspired_list[2] = {{kServableName, 1, 1}, {kServableName2, 0, 2}};
  for (const auto& servable_aspired : servable_aspired_list) {
    std::vector<ServableData<std::unique_ptr<Loader>>> aspired_versions;
    for (int i = servable_aspired.start; i <= servable_aspired.end; ++i) {
      const ServableId id = {servable_aspired.name.ToString(), i};
      aspired_versions.push_back(CreateAspiredVersion(id));
    }
    manager_->GetAspiredVersionsCallback()(servable_aspired.name,
                                           std::move(aspired_versions));
    HandlePendingAspiredVersionsRequests();
  }

  // The manager prefers to unload a servable before loading a servable, so it
  // should prefer to unload version 0 of the first servable stream.
  InvokePolicyAndExecuteAction();
  WaitUntilServableManagerStateIsOneOf(servable_state_monitor_,
                                       {kServableName, 0},
                                       {ServableState::ManagerState::kEnd});

  ServableHandle<int64> not_found_0_handle;
  const Status not_found_0_status = manager_->GetServableHandle(
      ServableRequest::Specific(kServableName, 0), &not_found_0_handle);
  ASSERT_FALSE(not_found_0_status.ok()) << not_found_0_status;
  EXPECT_EQ(error::NOT_FOUND, not_found_2_status.code());

  not_found_2_status = manager_->GetServableHandle(
      ServableRequest::Specific(kServableName2, 2), &not_found_2_handle);
  ASSERT_FALSE(not_found_2_status.ok()) << not_found_2_status;
  EXPECT_EQ(error::NOT_FOUND, not_found_2_status.code());

  // Now it should load version 2 of the second servable stream.
  InvokePolicyAndExecuteAction();
  WaitUntilServableManagerStateIsOneOf(
      servable_state_monitor_, {kServableName2, 2},
      {ServableState::ManagerState::kAvailable});

  ServableHandle<int64> found_2_handle;
  const Status found_2_status = manager_->GetServableHandle(
      ServableRequest::Specific(kServableName2, 2), &found_2_handle);
  TF_ASSERT_OK(found_2_status);
  EXPECT_EQ(2, *found_2_handle);
}

// Test to ensure the manager doesn't try to load or serve an incoming erroneous
// aspired-version entry.
TEST_P(AspiredVersionsManagerTest, ErroneousAspiredVersion) {
  std::vector<ServableData<std::unique_ptr<Loader>>> aspired_versions;
  aspired_versions.push_back(CreateErroneousAspiredVersion({kServableName, 3}));
  manager_->GetAspiredVersionsCallback()(kServableName,
                                         std::move(aspired_versions));
  HandlePendingAspiredVersionsRequests();

  ServableHandle<int64> handle;
  Status status = manager_->GetServableHandle(
      ServableRequest::Specific(kServableName, 3), &handle);
  EXPECT_FALSE(status.ok()) << status;

  InvokePolicyAndExecuteAction();

  status = manager_->GetServableHandle(
      ServableRequest::Specific(kServableName, 3), &handle);
  EXPECT_FALSE(status.ok()) << status;
}

// Test to ensure that the deletion of a loader/servable occurs in a manager
// thread, and not a request thread.
TEST_P(AspiredVersionsManagerTest, DestructOnNonServingThread) {
  std::unique_ptr<ServableHandle<int64>> latest_handle(
      new ServableHandle<int64>());
  const Status status = manager_->GetServableHandle(
      ServableRequest::Latest(kServableName), latest_handle.get());
  TF_ASSERT_OK(status);
  EXPECT_EQ(1, **latest_handle);

  manager_->GetAspiredVersionsCallback()(kServableName, {});
  HandlePendingAspiredVersionsRequests();

  Notification done_unload_servable;
  std::unique_ptr<Thread> unload_servable(
      Env::Default()->StartThread({}, "UnloadServable", [&]() {
        // Unload the servable.
        for (int i = 0; i < kNumVersionsPerServable; ++i) {
          InvokePolicyAndExecuteAction();
        }
        WaitUntilServableManagerStateIsOneOf(
            servable_state_monitor_, {kServableName, 0},
            {ServableState::ManagerState::kEnd});
        FlushServables();
        // The servable has been deleted in this thread if there is no
        // thread-pool for unload.
        if (thread_pool_sizes_.num_unload_threads == 0) {
          EXPECT_TRUE(FakeLoader::was_deleted_in_this_thread());
        }
        done_unload_servable.Notify();
      }));

  // This will unblock the UnloadServable.
  latest_handle.reset();
  done_unload_servable.WaitForNotification();
  // The servable wasn't deleted in this thread.
  EXPECT_FALSE(FakeLoader::was_deleted_in_this_thread());
}

MATCHER_P(EqualsServableState, servable_state, servable_state.DebugString()) {
  if (arg == servable_state) {
    return true;
  }
  *result_listener << arg.DebugString();
  return false;
}

TEST_P(AspiredVersionsManagerTest, EventBusErroneousVersion) {
  std::vector<ServableData<std::unique_ptr<Loader>>> aspired_versions;
  const ServableId id = {kServableName, 3};
  aspired_versions.push_back(
      ServableData<std::unique_ptr<Loader>>(id, errors::Unknown("error")));
  manager_->GetAspiredVersionsCallback()(kServableName,
                                         std::move(aspired_versions));
  HandlePendingAspiredVersionsRequests();

  const ServableState expected_published_state = {
      id, ServableState::ManagerState::kEnd, errors::Unknown("error")};
  EXPECT_THAT(*servable_state_monitor_.GetState(id),
              EqualsServableState(expected_published_state));
}

TEST_P(AspiredVersionsManagerTest, EventBusErrorOnLoad) {
  std::vector<ServableData<std::unique_ptr<Loader>>> aspired_versions;
  const ServableId id = {kServableName, 7};
  std::unique_ptr<Loader> loader(
      new FakeLoader(7, errors::Internal("Error on load.")));
  aspired_versions.push_back({id, std::move(loader)});
  manager_->GetAspiredVersionsCallback()(kServableName,
                                         std::move(aspired_versions));
  HandlePendingAspiredVersionsRequests();

  const ServableState start_state = {id, ServableState::ManagerState::kStart,
                                     Status::OK()};
  EXPECT_THAT(*servable_state_monitor_.GetState(id),
              EqualsServableState(start_state));

  // Unload version 0 and load the new aspired version. Version 1 may or may not
  // be unloaded (depending on whether load/unload thread pools are used).
  for (int i = 0; i < kNumVersionsPerServable + 1; ++i) {
    InvokePolicyAndExecuteAction();
  }
  WaitUntilServableManagerStateIsOneOf(servable_state_monitor_, id,
                                       {ServableState::ManagerState::kEnd});

  const ServableState error_state = {id, ServableState::ManagerState::kEnd,
                                     errors::Internal("Error on load.")};
  EXPECT_THAT(*servable_state_monitor_.GetState(id),
              EqualsServableState(error_state));
}

TEST_P(AspiredVersionsManagerTest, EventBusServableLifecycle) {
  std::vector<ServableData<std::unique_ptr<Loader>>> aspired_versions;
  const ServableId id = {kServableName, 7};
  test_util::MockLoader* loader = new NiceMock<test_util::MockLoader>();
  aspired_versions.push_back({id, std::unique_ptr<Loader>(loader)});
  manager_->GetAspiredVersionsCallback()(kServableName,
                                         std::move(aspired_versions));
  HandlePendingAspiredVersionsRequests();

  const ServableState start_state = {id, ServableState::ManagerState::kStart,
                                     Status::OK()};
  EXPECT_THAT(*servable_state_monitor_.GetState(id),
              EqualsServableState(start_state));

  Notification load_called;
  Notification load_continue;
  EXPECT_CALL(*loader, Load()).WillOnce(InvokeWithoutArgs([&]() {
    load_called.Notify();
    load_continue.WaitForNotification();
    return Status::OK();
  }));

  std::unique_ptr<Thread> load_unload_thread(
      Env::Default()->StartThread(ThreadOptions(), "LoadUnloadThread", [&]() {
        // Unload version 0 and load the new aspired version. Version 1 may or
        // may not be unloaded (depending on whether load/unload thread pools
        // are used).
        for (int i = 0; i < kNumVersionsPerServable + 1; ++i) {
          InvokePolicyAndExecuteAction();
        }
      }));

  load_called.WaitForNotification();

  const ServableState loading_state = {
      id, ServableState::ManagerState::kLoading, Status::OK()};
  EXPECT_THAT(*servable_state_monitor_.GetState(id),
              EqualsServableState(loading_state));

  load_continue.Notify();
  WaitUntilServableManagerStateIsOneOf(
      servable_state_monitor_, id, {ServableState::ManagerState::kAvailable});

  const ServableState available_state = {
      id, ServableState::ManagerState::kAvailable, Status::OK()};
  EXPECT_THAT(*servable_state_monitor_.GetState(id),
              EqualsServableState(available_state));

  manager_->GetAspiredVersionsCallback()(kServableName, {});
  HandlePendingAspiredVersionsRequests();

  Notification unload_called;
  Notification unload_continue;
  EXPECT_CALL(*loader, Unload()).WillOnce(Invoke([&]() {
    unload_called.Notify();
    unload_continue.WaitForNotification();
  }));

  std::unique_ptr<Thread> unload_thread(
      Env::Default()->StartThread(ThreadOptions(), "UnloadThread", [&]() {
        // Call InvokePolicyAndExecuteAction() twice to unload version 1 and the
        // new version, in case version 1 has not been unloaded previously.
        InvokePolicyAndExecuteAction();
        InvokePolicyAndExecuteAction();
      }));

  unload_called.WaitForNotification();

  const ServableState unloading_state = {
      id, ServableState::ManagerState::kUnloading, Status::OK()};
  EXPECT_THAT(*servable_state_monitor_.GetState(id),
              EqualsServableState(unloading_state));

  unload_continue.Notify();
  WaitUntilServableManagerStateIsOneOf(servable_state_monitor_, id,
                                       {ServableState::ManagerState::kEnd});

  const ServableState end_state = {
      {kServableName, 7}, ServableState::ManagerState::kEnd, Status::OK()};
  EXPECT_THAT(*servable_state_monitor_.GetState(id),
              EqualsServableState(end_state));
}

// Tests whether there are any errors if we don't have an event bus configured.
TEST_P(AspiredVersionsManagerTest, NoEventBus) {
  AspiredVersionsManager::Options options;
  // The state manager thread won't be run automatically.
  options.manage_state_interval_micros = -1;
  options.env = Env::Default();
  options.aspired_version_policy.reset(new AvailabilityPreservingPolicy());
  std::unique_ptr<AspiredVersionsManager> aspired_versions_manager;
  TF_ASSERT_OK(AspiredVersionsManager::Create(std::move(options),
                                              &aspired_versions_manager));

  std::vector<ServableData<std::unique_ptr<Loader>>> aspired_versions;
  const ServableId id = {kServableName, 7};
  std::unique_ptr<Loader> loader(new FakeLoader(7));
  aspired_versions.push_back({id, std::move(loader)});
  aspired_versions_manager->GetAspiredVersionsCallback()(
      kServableName, std::move(aspired_versions));
  HandlePendingAspiredVersionsRequests();
}

TEST_P(AspiredVersionsManagerTest, RetryOnLoadErrorFinallySucceeds) {
  CHECK_GE(max_num_load_retries_, 1);

  test_util::MockLoader* loader = new NiceMock<test_util::MockLoader>;
  // We succeed on the last load, before the manager gives up.
  EXPECT_CALL(*loader, Load())
      .WillOnce(Return(errors::Internal("Error on load.")))
      .WillOnce(Return(Status::OK()));

  const ServableId id = {kServableName, 7};
  std::vector<ServableData<std::unique_ptr<Loader>>> aspired_versions;
  aspired_versions.push_back({id, std::unique_ptr<Loader>(loader)});
  manager_->GetAspiredVersionsCallback()(kServableName,
                                         std::move(aspired_versions));
  HandlePendingAspiredVersionsRequests();

  // Unload version 0 and load the new aspired version. Version 1 may or may not
  // be unloaded (depending on whether load/unload thread pools are used).
  for (int i = 0; i < kNumVersionsPerServable + 1; ++i) {
    InvokePolicyAndExecuteAction();
  }
  WaitUntilServableManagerStateIsOneOf(
      servable_state_monitor_, id, {ServableState::ManagerState::kAvailable});

  const ServableState available_state = {
      id, ServableState::ManagerState::kAvailable, Status::OK()};
  EXPECT_THAT(*servable_state_monitor_.GetState(id),
              EqualsServableState(available_state));
}

TEST_P(AspiredVersionsManagerTest, RetryOnLoadErrorFinallyFails) {
  CHECK_GE(max_num_load_retries_, 1);

  std::vector<ServableData<std::unique_ptr<Loader>>> aspired_versions;
  const ServableId id = {kServableName, 7};
  // We always fail.
  std::unique_ptr<Loader> loader(
      new FakeLoader(7, errors::Internal("Error on load.")));
  aspired_versions.push_back({id, std::move(loader)});
  manager_->GetAspiredVersionsCallback()(kServableName,
                                         std::move(aspired_versions));
  HandlePendingAspiredVersionsRequests();

  // Unload version 0 and load the new aspired version. Version 1 may or may not
  // be unloaded (depending on whether load/unload thread pools are used).
  for (int i = 0; i < kNumVersionsPerServable + 1; ++i) {
    InvokePolicyAndExecuteAction();
  }
  WaitUntilServableManagerStateIsOneOf(servable_state_monitor_, id,
                                       {ServableState::ManagerState::kEnd});

  const ServableState error_state = {id, ServableState::ManagerState::kEnd,
                                     errors::Internal("Error on load.")};
  EXPECT_THAT(*servable_state_monitor_.GetState(id),
              EqualsServableState(error_state));
}

// Tests the interaction between AspiredVersionsManager and the
// AvailabilityPreservingPolicy.
// Specifically, we want to make sure that the manager will not try to unload
// all serving versions that are no longer aspired if the new aspired version
// was not able to start serving.
TEST_P(AspiredVersionsManagerTest, AspireErrorDontUnload) {
  const std::vector<ServableId> expected_before = {{kServableName, 0},
                                                   {kServableName, 1},
                                                   {kServableName2, 0},
                                                   {kServableName2, 1}};
  EXPECT_THAT(manager_->ListAvailableServableIds(),
              UnorderedElementsAreArray(expected_before));

  // Set stream kServableName to have servable 7.
  // This causes 0 & 1 to be set to not aspired and 7 to be loaded, but 7 errors
  // on load, so never moves to a loaded state.
  {
    std::vector<ServableData<std::unique_ptr<Loader>>> aspired_versions;
    const ServableId id = {kServableName, 7};
    std::unique_ptr<Loader> loader(
        new FakeLoader(7, errors::Internal("An error.")));
    aspired_versions.push_back({id, std::move(loader)});
    manager_->GetAspiredVersionsCallback()(kServableName,
                                           std::move(aspired_versions));
    HandlePendingAspiredVersionsRequests();

    // Will unload version 0.
    InvokePolicyAndExecuteAction();
    WaitUntilServableManagerStateIsOneOf(servable_state_monitor_,
                                         {kServableName, 0},
                                         {ServableState::ManagerState::kEnd});

    // Will try to load version 7 and fail.
    InvokePolicyAndExecuteAction();
    WaitUntilServableManagerStateIsOneOf(servable_state_monitor_, id,
                                         {ServableState::ManagerState::kEnd});
  }

  // For kServableName, version 0 has been unloaded. For kServableName2, both
  // versions should still be loaded.
  const std::vector<ServableId> expected_after_first_load = {
      {kServableName, 1}, {kServableName2, 0}, {kServableName2, 1}};
  EXPECT_THAT(manager_->ListAvailableServableIds(),
              UnorderedElementsAreArray(expected_after_first_load));

  // Now successfully loading a new version should allow the older versions to
  // be unloaded.
  {
    std::vector<ServableData<std::unique_ptr<Loader>>> aspired_versions;
    const ServableId id = {kServableName, 8};
    std::unique_ptr<Loader> loader(new FakeLoader(8));
    aspired_versions.push_back({id, std::move(loader)});
    manager_->GetAspiredVersionsCallback()(kServableName,
                                           std::move(aspired_versions));
    HandlePendingAspiredVersionsRequests();

    // Will try to load version 8 and succeed.
    InvokePolicyAndExecuteAction();
    WaitUntilServableManagerStateIsOneOf(
        servable_state_monitor_, id, {ServableState::ManagerState::kAvailable});

    // Will unload version 1.
    InvokePolicyAndExecuteAction();
    WaitUntilServableManagerStateIsOneOf(servable_state_monitor_,
                                         {kServableName, 1},
                                         {ServableState::ManagerState::kEnd});
  }
}

TEST_P(AspiredVersionsManagerTest, UnaspireThenImmediatelyReaspire) {
  // This test exercises a scenario in which a servable has been unaspired, and
  // while it is still being managed (e.g. loading, serving or unloading) it
  // gets reaspired (with a new loader). The manager should wait for the
  // original loader to get taken down via the normal process for unaspired
  // loaders, and then proceed to bring up the new loader.

  const ServableId id = {kServableName, 7};

  std::vector<ServableData<std::unique_ptr<Loader>>> first_aspired_versions;
  test_util::MockLoader* first_loader = new NiceMock<test_util::MockLoader>();
  first_aspired_versions.push_back({id, std::unique_ptr<Loader>(first_loader)});
  EXPECT_CALL(*first_loader, Load()).WillOnce(Return(Status::OK()));
  manager_->GetAspiredVersionsCallback()(kServableName,
                                         std::move(first_aspired_versions));
  HandlePendingAspiredVersionsRequests();

  // Wait for verion 0 to be unloaded and the new aspired version to be loaded.
  // If we don't wait, the first_loader_handle below may be obtained before
  // the loading or unloading finishes, which may block the loading or
  // unloading.
  InvokePolicyAndExecuteAction();
  InvokePolicyAndExecuteAction();
  WaitUntilServableManagerStateIsOneOf(
      servable_state_monitor_, id, {ServableState::ManagerState::kAvailable});
  WaitUntilServableManagerStateIsOneOf(servable_state_monitor_,
                                       {kServableName, 0},
                                       {ServableState::ManagerState::kEnd});

  // Pin 'first_loader' in the manager by holding a handle to its servable.
  int servable = 42;
  EXPECT_CALL(*first_loader, servable()).WillOnce(InvokeWithoutArgs([&]() {
    return AnyPtr{&servable};
  }));
  auto first_loader_handle =
      std::unique_ptr<ServableHandle<int>>(new ServableHandle<int>);
  TF_ASSERT_OK(manager_->GetServableHandle(ServableRequest::FromId(id),
                                           first_loader_handle.get()));

  // Now, we'll un-aspire the servable, and then re-aspire it with a new loader.
  // The manager should wait until it is able to unload the first loader, then
  // bring up the second loader.

  Notification first_unload_called;
  EXPECT_CALL(*first_loader, Unload()).WillOnce(InvokeWithoutArgs([&]() {
    first_unload_called.Notify();
  }));

  std::vector<ServableData<std::unique_ptr<Loader>>> empty_aspired_versions;
  manager_->GetAspiredVersionsCallback()(kServableName,
                                         std::move(empty_aspired_versions));
  HandlePendingAspiredVersionsRequests();

  // The following thread will block trying to unload the first loader, while we
  // hold the handle.
  std::unique_ptr<Thread> unload_thread(
      Env::Default()->StartThread(ThreadOptions(), "UnloadThread", [&]() {
        // Unload version 1 and the newly un-aspired version.
        InvokePolicyAndExecuteAction();
        InvokePolicyAndExecuteAction();
      }));

  // Re-aspire the servable with a fresh loader.
  std::vector<ServableData<std::unique_ptr<Loader>>> second_aspired_versions;
  test_util::MockLoader* second_loader = new NiceMock<test_util::MockLoader>();
  second_aspired_versions.push_back(
      {id, std::unique_ptr<Loader>(second_loader)});
  Notification second_load_called;
  EXPECT_CALL(*second_loader, Load()).WillOnce(InvokeWithoutArgs([&]() {
    second_load_called.Notify();
    return Status::OK();
  }));
  manager_->GetAspiredVersionsCallback()(kServableName,
                                         std::move(second_aspired_versions));

  // Run the manager's background logic in a loop. Nothing should happen for now
  // because the first loader is pinned.
  std::unique_ptr<Thread> reaspire_thread(
      Env::Default()->StartThread(ThreadOptions(), "ReaspireThread", [&]() {
        while (!second_load_called.HasBeenNotified()) {
          FlushServables();
          HandlePendingAspiredVersionsRequests();
          InvokePolicyAndExecuteAction();
          Env::Default()->SleepForMicroseconds(1000 /* 1 ms */);
        }
      }));
  Env::Default()->SleepForMicroseconds(50 * 1000 /* 50 ms */);
  EXPECT_FALSE(first_unload_called.HasBeenNotified());
  EXPECT_FALSE(second_load_called.HasBeenNotified());

  // Unpin the first loader. The manager should unload the first loader and
  // bring up the second loader.
  first_loader_handle = nullptr;
  first_unload_called.WaitForNotification();
  second_load_called.WaitForNotification();
}

TEST_P(AspiredVersionsManagerTest,
       UnaspireFailedServableThenImmediatelyReaspire) {
  // Like UnaspireThenImmediatelyReaspire, but covers the case in which the
  // servable fails to load the first time it is aspired.

  const ServableId id = {kServableName, 7};

  std::vector<ServableData<std::unique_ptr<Loader>>> first_aspired_versions;
  test_util::MockLoader* first_loader = new NiceMock<test_util::MockLoader>();
  first_aspired_versions.push_back({id, std::unique_ptr<Loader>(first_loader)});
  EXPECT_CALL(*first_loader, Load())
      .WillRepeatedly(Return(Status(error::UNKNOWN, "first load failing")));
  manager_->GetAspiredVersionsCallback()(kServableName,
                                         std::move(first_aspired_versions));
  HandlePendingAspiredVersionsRequests();
  // Unload version 0 and load the new aspired version. Version 1 may or may not
  // be unloaded (depending on whether load/unload thread pools are used).
  for (int i = 0; i < kNumVersionsPerServable + 1; ++i) {
    InvokePolicyAndExecuteAction();
  }
  WaitUntilServableManagerStateIsOneOf(servable_state_monitor_, id,
                                       {ServableState::ManagerState::kEnd});

  // Now, we'll un-aspire the servable, and then re-aspire it with a new loader.
  // The manager should wait until it is able to flush the first loader, then
  // bring up the second loader.

  std::vector<ServableData<std::unique_ptr<Loader>>> empty_aspired_versions;
  manager_->GetAspiredVersionsCallback()(kServableName,
                                         std::move(empty_aspired_versions));
  HandlePendingAspiredVersionsRequests();

  // Re-aspire the servable with a fresh loader.
  std::vector<ServableData<std::unique_ptr<Loader>>> second_aspired_versions;
  test_util::MockLoader* second_loader = new NiceMock<test_util::MockLoader>();
  second_aspired_versions.push_back(
      {id, std::unique_ptr<Loader>(second_loader)});
  Notification second_load_called;
  EXPECT_CALL(*second_loader, Load()).WillOnce(InvokeWithoutArgs([&]() {
    second_load_called.Notify();
    return Status::OK();
  }));
  manager_->GetAspiredVersionsCallback()(kServableName,
                                         std::move(second_aspired_versions));

  // Run the manager's background logic in a loop, but sans FlushServables().
  // Nothing should happen for now because the first loader isn't flushed.
  std::unique_ptr<Thread> reaspire_thread(
      Env::Default()->StartThread(ThreadOptions(), "ReaspireThread", [&]() {
        while (!second_load_called.HasBeenNotified()) {
          HandlePendingAspiredVersionsRequests();
          InvokePolicyAndExecuteAction();
          Env::Default()->SleepForMicroseconds(1000 /* 1 ms */);
        }
      }));
  Env::Default()->SleepForMicroseconds(50 * 1000 /* 50 ms */);
  EXPECT_FALSE(second_load_called.HasBeenNotified());

  // Flush the first loader. The manager should finally bring up the second
  // loader.
  FlushServables();
  second_load_called.WaitForNotification();
}

TEST_P(AspiredVersionsManagerTest, UnaspireNewServableThenImmediatelyReaspire) {
  // Like UnaspireThenImmediatelyReaspire, but covers the case in which the
  // servable is in state kNew when it gets unaspired.
  // (Regression test for b/27766674.)

  const ServableId id = {kServableName, 7};

  std::vector<ServableData<std::unique_ptr<Loader>>> first_aspired_versions;
  test_util::MockLoader* first_loader = new NiceMock<test_util::MockLoader>();
  EXPECT_CALL(*first_loader, Load()).Times(0);
  first_aspired_versions.push_back({id, std::unique_ptr<Loader>(first_loader)});
  manager_->GetAspiredVersionsCallback()(kServableName,
                                         std::move(first_aspired_versions));
  HandlePendingAspiredVersionsRequests();
  // (We *don't* call InvokePolicyAndExecuteAction(), thus causing the servable
  // to remain in state kNew.)

  // Now, we'll un-aspire the servable, and then re-aspire it with a new loader.
  // The manager should get rid of the first loader, then bring up the second
  // one.

  std::vector<ServableData<std::unique_ptr<Loader>>> empty_aspired_versions;
  manager_->GetAspiredVersionsCallback()(kServableName,
                                         std::move(empty_aspired_versions));
  HandlePendingAspiredVersionsRequests();

  // Re-aspire the servable with a fresh loader.
  std::vector<ServableData<std::unique_ptr<Loader>>> second_aspired_versions;
  test_util::MockLoader* second_loader = new NiceMock<test_util::MockLoader>();
  second_aspired_versions.push_back(
      {id, std::unique_ptr<Loader>(second_loader)});
  Notification second_load_called;
  EXPECT_CALL(*second_loader, Load()).WillOnce(InvokeWithoutArgs([&]() {
    second_load_called.Notify();
    return Status::OK();
  }));
  manager_->GetAspiredVersionsCallback()(kServableName,
                                         std::move(second_aspired_versions));
  // The first HandlePendingAspiredVersionsRequests() call will do nothing,
  // because the first loader remains in the manager (with state kNew).
  HandlePendingAspiredVersionsRequests();
  // FlushServables() should remove the first loader, thus clearing the way for
  // a subsequent HandlePendingAspiredVersionsRequests() call to accept the
  // second loader.
  FlushServables();
  HandlePendingAspiredVersionsRequests();
  // Unload version 0 and load the new aspired version. Version 1 may or may not
  // be unloaded (depending on whether load/unload thread pools are used).
  for (int i = 0; i < kNumVersionsPerServable + 1; ++i) {
    InvokePolicyAndExecuteAction();
  }
  second_load_called.WaitForNotification();
}

class MockAspiredVersionPolicy : public AspiredVersionPolicy {
 public:
  MOCK_CONST_METHOD1(GetNextAction,
                     optional<ServableAction>(
                         const std::vector<AspiredServableStateSnapshot>&));
};

TEST(AspiredVersionsManagerTest, CallPolicyWithAllVersions) {
  std::unique_ptr<AspiredVersionsManager> manager;
  AspiredVersionsManager::Options manager_options;
  MockAspiredVersionPolicy* policy = new MockAspiredVersionPolicy;
  // The state manager thread won't be run automatically.
  manager_options.manage_state_interval_micros = -1;
  manager_options.aspired_version_policy =
      std::unique_ptr<AspiredVersionPolicy>(policy);
  TF_CHECK_OK(
      AspiredVersionsManager::Create(std::move(manager_options), &manager));
  std::set<ServableId> servables;
  std::vector<ServableData<std::unique_ptr<Loader>>> aspired_versions;
  for (int i = 0; i < kNumVersionsPerServable; ++i) {
    const ServableId id = {kServableName, i};
    aspired_versions.push_back(CreateAspiredVersion(id));
    servables.insert(id);
  }
  manager->GetAspiredVersionsCallback()(kServableName,
                                        std::move(aspired_versions));
  test_util::AspiredVersionsManagerTestAccess(manager.get())
      .HandlePendingAspiredVersionsRequests();

  std::vector<AspiredServableStateSnapshot> all_versions;
  EXPECT_CALL(*policy, GetNextAction(_))
      .WillOnce(Invoke([&all_versions](
          const std::vector<AspiredServableStateSnapshot>& snapshots) {
        all_versions = snapshots;
        return nullopt;
      }));
  test_util::AspiredVersionsManagerTestAccess(manager.get())
      .InvokePolicyAndExecuteAction();
  EXPECT_EQ(kNumVersionsPerServable, all_versions.size());
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
