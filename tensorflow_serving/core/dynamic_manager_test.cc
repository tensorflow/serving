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

#include "tensorflow_serving/core/dynamic_manager.h"

#include <algorithm>
#include <functional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow_serving/core/eager_load_policy.h"
#include "tensorflow_serving/core/servable_state.h"
#include "tensorflow_serving/core/test_util/dynamic_manager_test_util.h"
#include "tensorflow_serving/core/test_util/fake_loader.h"
#include "tensorflow_serving/core/test_util/mock_loader.h"
#include "tensorflow_serving/util/any_ptr.h"
#include "tensorflow_serving/util/event_bus.h"

namespace tensorflow {
namespace serving {

using ::testing::_;
using ::testing::Invoke;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::UnorderedElementsAreArray;
using test_util::FakeLoader;

namespace {

constexpr char kServableName[] = "kServableName";
constexpr char kServableName2[] = "kServableName2";
constexpr int kNumVersionsPerServable = 2;
constexpr int kNumTotalVersions = 4;

class DynamicManagerTest : public ::testing::Test {
 protected:
  DynamicManagerTest()
      : servable_event_bus_(EventBus<ServableState>::CreateEventBus()) {
    servable_state_subscription_ =
        servable_event_bus_->Subscribe([this](const ServableState& state) {
          LOG(INFO) << "Published state: " << state.DebugString();
          last_published_servable_state_ = state;
        });
    // The state manager thread won't be run automatically.
    dynamic_manager_options_.manage_state_interval_micros = -1;
    dynamic_manager_options_.env = Env::Default();
    dynamic_manager_options_.version_policy.reset(new EagerLoadPolicy());
    dynamic_manager_options_.servable_event_bus = servable_event_bus_.get();
    dynamic_manager_options_.max_num_load_tries = 2;
    dynamic_manager_options_.load_retry_interval_micros = 0;
    // dynamic_manager_options_.load_retry_interval_micros = 0;
    manager_.reset(new DynamicManager(std::move(dynamic_manager_options_)));
  }

  // Creates an aspired-versions entry with 'id' and a FakeLoader whose servable
  // is id.version.
  ServableData<std::unique_ptr<Loader>> CreateAspiredVersion(
      const ServableId& id) {
    std::unique_ptr<Loader> loader(new FakeLoader(id.version));
    return CreateServableData(id, std::move(loader));
  }

  // Creates an aspired-versions entry with 'id' and an error (and no loader).
  ServableData<std::unique_ptr<Loader>> CreateErroneousAspiredVersion(
      const ServableId& id) {
    return ServableData<std::unique_ptr<Loader>>(id, errors::Unknown("error"));
  }

  void SetUp() override {
    // We setUp the manager_ with two different servable streams, each with two
    // aspired versions 0 and 1.
    std::vector<ServableData<std::unique_ptr<Loader>>> aspired_versions;
    for (int i = 0; i < kNumVersionsPerServable; ++i) {
      aspired_versions.push_back(CreateAspiredVersion({kServableName, i}));
    }
    manager_->GetAspiredVersionsCallback()(kServableName,
                                           std::move(aspired_versions));

    std::vector<ServableData<std::unique_ptr<Loader>>> aspired_versions2;
    for (int i = 0; i < kNumVersionsPerServable; ++i) {
      aspired_versions2.push_back(CreateAspiredVersion({kServableName2, i}));
    }
    manager_->GetAspiredVersionsCallback()(kServableName2,
                                           std::move(aspired_versions2));

    for (int i = 0; i < kNumTotalVersions; ++i) {
      // Each time the state manager thread is run, we should load a servable
      // version.
      RunManageState();
    }
  }

  void RunManageState() {
    test_util::DynamicManagerTestAccess(manager_.get()).RunManageState();
  }

  std::shared_ptr<EventBus<ServableState>> servable_event_bus_;
  std::unique_ptr<EventBus<ServableState>::Subscription>
      servable_state_subscription_;
  ServableState last_published_servable_state_;
  DynamicManager::Options dynamic_manager_options_;
  std::unique_ptr<DynamicManager> manager_;
};

TEST_F(DynamicManagerTest, ServableHandleNotFoundMissingLoaderName) {
  ServableHandle<int64> handle;
  const Status status = manager_->GetServableHandle(
      ServableRequest::Latest(strings::StrCat(kServableName, "missing")),
      &handle);
  ASSERT_FALSE(status.ok()) << status;
  EXPECT_EQ(error::NOT_FOUND, status.code());
}

TEST_F(DynamicManagerTest, ServableHandleNotFoundMissingVersion) {
  // This version is missing.
  const int64 missing_version = 100;
  ServableHandle<int64> handle;
  const Status status = manager_->GetServableHandle(
      ServableRequest::Specific(kServableName, missing_version), &handle);
  ASSERT_FALSE(status.ok()) << status;
  EXPECT_EQ(error::NOT_FOUND, status.code());
}

TEST_F(DynamicManagerTest, ServableHandleInvalidArgument) {
  // The servable is supposed to be an int type and we ask for a float type,
  // thus causing an invalid argument error.
  ServableHandle<float> handle;
  const Status status = manager_->GetServableHandle(
      ServableRequest::Latest(kServableName), &handle);
  ASSERT_FALSE(status.ok()) << status;
  EXPECT_EQ(error::INVALID_ARGUMENT, status.code());
}

TEST_F(DynamicManagerTest, ServableHandleLatest) {
  std::vector<ServableData<std::unique_ptr<Loader>>> aspired_versions;
  aspired_versions.push_back(
      CreateAspiredVersion({kServableName, kNumVersionsPerServable + 1}));
  manager_->GetAspiredVersionsCallback()(kServableName,
                                         std::move(aspired_versions));
  RunManageState();

  ServableHandle<int64> handle;
  const Status status = manager_->GetServableHandle(
      ServableRequest::Latest(kServableName), &handle);
  TF_ASSERT_OK(status);
  EXPECT_EQ(kNumVersionsPerServable + 1, *handle);
}

// Test the case where the latest version of a servable available is 0.
TEST_F(DynamicManagerTest, ServableHandleLatestVersionIsZero) {
  const char kServableName3[] = "kServableName3";

  std::vector<ServableData<std::unique_ptr<Loader>>> aspired_versions;
  aspired_versions.push_back(CreateAspiredVersion({kServableName3, 0}));
  manager_->GetAspiredVersionsCallback()(kServableName3,
                                         std::move(aspired_versions));
  RunManageState();

  ServableHandle<int64> handle;
  const Status status = manager_->GetServableHandle(
      ServableRequest::Latest(kServableName3), &handle);
  TF_ASSERT_OK(status);
  EXPECT_EQ(0, *handle);
}

TEST_F(DynamicManagerTest, ServableHandleSpecificVersion) {
  ServableHandle<int64> handle;
  const Status status = manager_->GetServableHandle(
      ServableRequest::Specific(kServableName2, 0), &handle);
  TF_ASSERT_OK(status);
  EXPECT_EQ(0, *handle);
}

TEST_F(DynamicManagerTest, ListAvailableServableIds) {
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
      new FakeLoader(7, errors::Internal("Error on load.")));
  aspired_versions.push_back({id, std::move(loader)});
  manager_->GetAspiredVersionsCallback()(kServableName,
                                         std::move(aspired_versions));
  // We have to run the threads twice for unloading, because unloading of
  // quiesced servables happens in the next manager thread run.
  for (int i = 0; i < 2 * kNumVersionsPerServable; ++i) {
    RunManageState();
  }
  const std::vector<ServableId> expected_after = {{kServableName2, 0},
                                                  {kServableName2, 1}};
  EXPECT_THAT(manager_->ListAvailableServableIds(),
              UnorderedElementsAreArray(expected_after));
}

TEST_F(DynamicManagerTest, GetAvailableServableHandles) {
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
  // We have to run the threads twice for unloading, because unloading of
  // quiesced servables happens in the next manager thread run.
  for (int i = 0; i < 2 * kNumVersionsPerServable; ++i) {
    RunManageState();
  }

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

TEST_F(DynamicManagerTest, GetAvailableServableHandlesWrongType) {
  const std::map<ServableId, ServableHandle<int>> wrong_type_handles =
      manager_->GetAvailableServableHandles<int>();
  EXPECT_EQ(0, wrong_type_handles.size());
}

TEST_F(DynamicManagerTest, AspiredRemovedFull) {
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

  int num_fake_loaders_before = FakeLoader::num_fake_loaders();
  // We have to run the threads twice for unloading, because unloading of
  // quiesced servables happens in the next manager thread run.
  for (int i = 0; i < 2 * kNumVersionsPerServable; ++i) {
    RunManageState();
  }
  int num_fake_loaders_after = FakeLoader::num_fake_loaders();
  EXPECT_EQ(kNumVersionsPerServable,
            num_fake_loaders_before - num_fake_loaders_after);

  ServableHandle<int64> missing_handle;
  const Status missing_status = manager_->GetServableHandle(
      ServableRequest::Latest(kServableName), &missing_handle);
  ASSERT_FALSE(missing_status.ok());
  EXPECT_EQ(error::NOT_FOUND, missing_status.code());
}

TEST_F(DynamicManagerTest, AspiredRemovedPartial) {
  {
    ServableHandle<int64> handle;
    const Status status = manager_->GetServableHandle(
        ServableRequest::Specific(kServableName, 1), &handle);
    TF_ASSERT_OK(status);
    EXPECT_EQ(1, *handle);
  }

  std::vector<ServableData<std::unique_ptr<Loader>>> aspired_versions;
  aspired_versions.push_back(CreateAspiredVersion({kServableName, 0}));
  manager_->GetAspiredVersionsCallback()(kServableName,
                                         std::move(aspired_versions));

  RunManageState();

  ServableHandle<int64> missing_handle;
  const Status missing_status = manager_->GetServableHandle(
      ServableRequest::Specific(kServableName, 1), &missing_handle);
  ASSERT_FALSE(missing_status.ok());
  EXPECT_EQ(error::NOT_FOUND, missing_status.code());
}

TEST_F(DynamicManagerTest, AspiredAndManageStateLoad) {
  ServableHandle<int64> not_found_handle;
  const Status not_found_status = manager_->GetServableHandle(
      ServableRequest::Specific(kServableName, 2), &not_found_handle);
  ASSERT_FALSE(not_found_status.ok()) << not_found_status;
  EXPECT_EQ(error::NOT_FOUND, not_found_status.code());

  std::vector<ServableData<std::unique_ptr<Loader>>> aspired_versions;
  aspired_versions.push_back(CreateAspiredVersion({kServableName, 2}));
  manager_->GetAspiredVersionsCallback()(kServableName,
                                         std::move(aspired_versions));

  ServableHandle<int64> not_ready_handle;
  const Status not_ready_status = manager_->GetServableHandle(
      ServableRequest::Specific(kServableName, 2), &not_ready_handle);
  ASSERT_FALSE(not_ready_status.ok()) << not_ready_status;
  EXPECT_EQ(error::NOT_FOUND, not_ready_status.code());

  RunManageState();

  ServableHandle<int64> handle;
  const Status status = manager_->GetServableHandle(
      ServableRequest::Specific(kServableName, 2), &handle);
  TF_ASSERT_OK(status);
  EXPECT_EQ(2, *handle);
}

TEST_F(DynamicManagerTest, AspiredAndManageStateUnload) {
  {
    ServableHandle<int64> handle;
    const Status status = manager_->GetServableHandle(
        ServableRequest::Specific(kServableName, 0), &handle);
    TF_ASSERT_OK(status);
    EXPECT_EQ(0, *handle);
  }

  manager_->GetAspiredVersionsCallback()(kServableName, {});

  for (int i = 0; i < kNumVersionsPerServable; ++i) {
    RunManageState();
  }

  ServableHandle<int64> not_found_handle;
  const Status not_found_status = manager_->GetServableHandle(
      ServableRequest::Specific(kServableName, 2), &not_found_handle);
  ASSERT_FALSE(not_found_status.ok()) << not_found_status;
  EXPECT_EQ(error::NOT_FOUND, not_found_status.code());
}

// The manager prefers unloading over loading when deciding between different
// servable actions. This behaviour is tested here.
TEST_F(DynamicManagerTest, ManagerPrefersUnloadOverLoad) {
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
  }

  // The manager prefers to unload a servable before loading a servable, so it
  // should prefer to unload version 0 of the first servable stream.
  RunManageState();
  // We have to run the threads twice for unloading, because unloading of
  // quiesced servables happens in the next manager thread run.
  RunManageState();

  ServableHandle<int64> not_found_0_handle;
  const Status not_found_0_status = manager_->GetServableHandle(
      ServableRequest::Specific(kServableName, 0), &not_found_0_handle);
  ASSERT_FALSE(not_found_0_status.ok()) << not_found_0_status;
  EXPECT_EQ(error::NOT_FOUND, not_found_2_status.code());

  not_found_2_status = manager_->GetServableHandle(
      ServableRequest::Specific(kServableName, 0), &not_found_2_handle);
  ASSERT_FALSE(not_found_2_status.ok()) << not_found_2_status;
  EXPECT_EQ(error::NOT_FOUND, not_found_2_status.code());

  // Now it should load version 2 of the second servable stream.
  RunManageState();

  ServableHandle<int64> found_2_handle;
  const Status found_2_status = manager_->GetServableHandle(
      ServableRequest::Specific(kServableName2, 2), &found_2_handle);
  TF_ASSERT_OK(found_2_status);
  EXPECT_EQ(2, *found_2_handle);
}

// Test to ensure the manager doesn't try to load or serve an incoming erroneous
// aspired-version entry.
TEST_F(DynamicManagerTest, ErroneousAspiredVersion) {
  std::vector<ServableData<std::unique_ptr<Loader>>> aspired_versions;
  aspired_versions.push_back(CreateErroneousAspiredVersion({kServableName, 3}));
  manager_->GetAspiredVersionsCallback()(kServableName,
                                         std::move(aspired_versions));

  ServableHandle<int64> handle;
  Status status = manager_->GetServableHandle(
      ServableRequest::Specific(kServableName, 3), &handle);
  EXPECT_FALSE(status.ok()) << status;

  RunManageState();

  status = manager_->GetServableHandle(
      ServableRequest::Specific(kServableName, 3), &handle);
  EXPECT_FALSE(status.ok()) << status;
}

// Test to ensure that the deletion of a loader/servable occurs in a manager
// thread, and not a request thread.
TEST_F(DynamicManagerTest, DestructOnNonServingThread) {
  std::unique_ptr<ServableHandle<int64>> first_handle(
      new ServableHandle<int64>());
  const Status status = manager_->GetServableHandle(
      ServableRequest::Latest(kServableName), first_handle.get());
  TF_ASSERT_OK(status);
  EXPECT_EQ(1, **first_handle);

  int num_fake_loaders_before = FakeLoader::num_fake_loaders();
  manager_->GetAspiredVersionsCallback()(kServableName, {});

  Notification done_first_run_do_policy;
  std::unique_ptr<Thread> run_policy_1_(Env::Default()->StartThread(
      {}, "RunManageState1",
      [&]() {
        // This RunManageState updates the available servable ids.  This will
        // block until we delete the first_handle.
        RunManageState();
        // Nothing has been deleted.
        ASSERT_EQ(0, num_fake_loaders_before - FakeLoader::num_fake_loaders());
        done_first_run_do_policy.Notify();
      }));

  first_handle.reset();
  // Nothing has been deleted.
  ASSERT_EQ(0, num_fake_loaders_before - FakeLoader::num_fake_loaders());
  // This will unblock the RunManageState.
  done_first_run_do_policy.WaitForNotification();

  // We have to run the threads twice for unloading, because unloading of
  // quiesced servables happens in the next manager thread run.
  std::unique_ptr<Thread> run_policy_2_(Env::Default()->StartThread(
      {}, "RunManageState2",
      [&]() {
        ASSERT_EQ(0, num_fake_loaders_before - FakeLoader::num_fake_loaders());
        // Unloads and deletes the loader this time.
        RunManageState();
        // A loader has been deleted in this thread.
        EXPECT_EQ(1, num_fake_loaders_before - FakeLoader::num_fake_loaders());
      }));
}

MATCHER_P(EqualsServableState, servable_state, servable_state.DebugString()) {
  if (arg == servable_state) {
    return true;
  }
  *result_listener << arg.DebugString();
  return false;
}

TEST_F(DynamicManagerTest, EventBusErroneousVersion) {
  std::vector<ServableData<std::unique_ptr<Loader>>> aspired_versions;
  aspired_versions.push_back(CreateErroneousAspiredVersion({kServableName, 3}));
  manager_->GetAspiredVersionsCallback()(kServableName,
                                         std::move(aspired_versions));

  const ServableState expected_published_state = {
      {kServableName, 3},
      ServableState::ManagerState::kEnd,
      errors::Unknown("error")};
  EXPECT_THAT(last_published_servable_state_,
              EqualsServableState(expected_published_state));
}

TEST_F(DynamicManagerTest, EventBusErrorOnLoad) {
  std::vector<ServableData<std::unique_ptr<Loader>>> aspired_versions;
  const ServableId id = {kServableName, 7};
  std::unique_ptr<Loader> loader(
      new FakeLoader(7, errors::Internal("Error on load.")));
  aspired_versions.push_back({id, std::move(loader)});
  manager_->GetAspiredVersionsCallback()(kServableName,
                                         std::move(aspired_versions));

  const ServableState start_state = {{kServableName, 7},
                                     ServableState::ManagerState::kStart,
                                     Status::OK()};
  EXPECT_THAT(last_published_servable_state_, EqualsServableState(start_state));

  RunManageState();

  const ServableState error_state = {{kServableName, 7},
                                     ServableState::ManagerState::kEnd,
                                     errors::Internal("Error on load.")};
  EXPECT_THAT(last_published_servable_state_, EqualsServableState(error_state));
}

TEST_F(DynamicManagerTest, EventBusServableLifecycle) {
  std::vector<ServableData<std::unique_ptr<Loader>>> aspired_versions;
  const ServableId id = {kServableName, 7};
  std::unique_ptr<Loader> loader(new FakeLoader(7));
  aspired_versions.push_back({id, std::move(loader)});
  manager_->GetAspiredVersionsCallback()(kServableName,
                                         std::move(aspired_versions));

  const ServableState start_state = {{kServableName, 7},
                                     ServableState::ManagerState::kStart,
                                     Status::OK()};
  EXPECT_THAT(last_published_servable_state_, EqualsServableState(start_state));

  RunManageState();

  const ServableState available_state = {
      {kServableName, 7},
      ServableState::ManagerState::kAvailable,
      Status::OK()};
  EXPECT_THAT(last_published_servable_state_,
              EqualsServableState(available_state));

  manager_->GetAspiredVersionsCallback()(kServableName, {});

  // No state change should happen at this point.
  EXPECT_THAT(last_published_servable_state_,
              EqualsServableState(available_state));

  RunManageState();

  const ServableState unloading_state = {
      {kServableName, 7},
      ServableState::ManagerState::kUnloading,
      Status::OK()};
  EXPECT_THAT(last_published_servable_state_,
              EqualsServableState(unloading_state));

  RunManageState();

  const ServableState end_state = {{kServableName, 7},
                                   ServableState::ManagerState::kEnd,
                                   Status::OK()};
  EXPECT_THAT(last_published_servable_state_, EqualsServableState(end_state));
}

// Tests whether there are any errors if we don't have an event bus configured.
TEST_F(DynamicManagerTest, NoEventBus) {
  DynamicManager::Options options;
  // The state manager thread won't be run automatically.
  options.manage_state_interval_micros = -1;
  options.env = Env::Default();
  options.version_policy.reset(new EagerLoadPolicy());
  DynamicManager manager(std::move(options));

  std::vector<ServableData<std::unique_ptr<Loader>>> aspired_versions;
  const ServableId id = {kServableName, 7};
  std::unique_ptr<Loader> loader(new FakeLoader(7));
  aspired_versions.push_back({id, std::move(loader)});
  manager_->GetAspiredVersionsCallback()(kServableName,
                                         std::move(aspired_versions));
}

TEST_F(DynamicManagerTest, RetryOnLoadErrorFinallySucceeds) {
  std::vector<ServableData<std::unique_ptr<Loader>>> aspired_versions;

  test_util::MockLoader* loader = new NiceMock<test_util::MockLoader>;
  // Prevents it being changed without our knowledge.
  CHECK_EQ(dynamic_manager_options_.max_num_load_tries, 2);
  // We succeed on the last load, before the manager gives up.
  EXPECT_CALL(*loader, Load(_))
      .WillOnce(Return(errors::Internal("Error on load.")))
      .WillOnce(Return(Status::OK()));

  const ServableId id = {kServableName, 7};
  aspired_versions.push_back({id, std::unique_ptr<Loader>(loader)});
  manager_->GetAspiredVersionsCallback()(kServableName,
                                         std::move(aspired_versions));

  RunManageState();

  const ServableState available_state = {
      {kServableName, 7},
      ServableState::ManagerState::kAvailable,
      Status::OK()};
  EXPECT_THAT(last_published_servable_state_,
              EqualsServableState(available_state));
}

TEST_F(DynamicManagerTest, RetryOnLoadErrorFinallyFails) {
  std::vector<ServableData<std::unique_ptr<Loader>>> aspired_versions;
  const ServableId id = {kServableName, 7};
  // We always fail.
  std::unique_ptr<Loader> loader(
      new FakeLoader(7, errors::Internal("Error on load.")));
  aspired_versions.push_back({id, std::move(loader)});
  manager_->GetAspiredVersionsCallback()(kServableName,
                                         std::move(aspired_versions));

  RunManageState();

  const ServableState error_state = {{kServableName, 7},
                                     ServableState::ManagerState::kEnd,
                                     errors::Internal("Error on load.")};
  EXPECT_THAT(last_published_servable_state_, EqualsServableState(error_state));
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
