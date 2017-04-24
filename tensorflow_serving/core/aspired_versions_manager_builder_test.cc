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

#include "tensorflow_serving/core/aspired_versions_manager_builder.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow_serving/core/availability_preserving_policy.h"
#include "tensorflow_serving/core/servable_data.h"
#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/core/servable_state_monitor.h"
#include "tensorflow_serving/core/simple_loader.h"
#include "tensorflow_serving/core/storage_path.h"
#include "tensorflow_serving/core/test_util/availability_test_util.h"
#include "tensorflow_serving/core/test_util/fake_loader_source_adapter.h"
#include "tensorflow_serving/core/test_util/fake_storage_path_source_adapter.h"
#include "tensorflow_serving/util/event_bus.h"

namespace tensorflow {
namespace serving {
namespace {

using ::testing::ElementsAre;
using test_util::FakeLoaderSourceAdapter;
using test_util::FakeStoragePathSourceAdapter;
using test_util::WaitUntilServableManagerStateIsOneOf;

class AspiredVersionsManagerBuilderTest : public ::testing::Test {
 protected:
  AspiredVersionsManagerBuilderTest()
      : servable_event_bus_(EventBus<ServableState>::CreateEventBus()),
        servable_state_monitor_(servable_event_bus_.get()) {
    AspiredVersionsManagerBuilder::Options manager_options;
    manager_options.servable_event_bus = servable_event_bus_.get();
    manager_options.aspired_version_policy.reset(
        new AvailabilityPreservingPolicy());
    TF_CHECK_OK(AspiredVersionsManagerBuilder::Create(
        std::move(manager_options), &builder_));
  }

  std::unique_ptr<AspiredVersionsManagerBuilder> builder_;
  std::shared_ptr<EventBus<ServableState>> servable_event_bus_;
  ServableStateMonitor servable_state_monitor_;
};

TEST_F(AspiredVersionsManagerBuilderTest, AddSourceConnection) {
  auto* const adapter = new FakeLoaderSourceAdapter("adapter");
  builder_->AddSource(std::unique_ptr<FakeLoaderSourceAdapter>(adapter));
  std::unique_ptr<Manager> manager = builder_->Build();

  const ServableId id = {"servable", 1};
  adapter->SetAspiredVersions(
      id.name, {CreateServableData(id, StoragePath("/storage/path"))});
  WaitUntilServableManagerStateIsOneOf(
      servable_state_monitor_, id, {ServableState::ManagerState::kAvailable});
}

TEST_F(AspiredVersionsManagerBuilderTest, AddSourceChainConnection) {
  auto* const adapter0 = new FakeStoragePathSourceAdapter("adapter0");
  auto adapter1 = std::unique_ptr<FakeStoragePathSourceAdapter>(
      new FakeStoragePathSourceAdapter("adapter1"));
  auto adapter2 = std::unique_ptr<FakeLoaderSourceAdapter>(
      new FakeLoaderSourceAdapter("adapter2"));
  builder_->AddSourceChain(
      std::unique_ptr<FakeStoragePathSourceAdapter>(adapter0),
      std::move(adapter1), std::move(adapter2));
  std::unique_ptr<Manager> manager = builder_->Build();

  const ServableId id = {"servable", 1};
  adapter0->SetAspiredVersions(
      id.name, {CreateServableData(id, StoragePath("/storage/path"))});
  WaitUntilServableManagerStateIsOneOf(
      servable_state_monitor_, id, {ServableState::ManagerState::kAvailable});

  ServableHandle<StoragePath> handle;
  const Status status =
      manager->GetServableHandle(ServableRequest::FromId(id), &handle);
  EXPECT_EQ(StoragePath("/storage/path/adapter0/adapter1/adapter2"), *handle);
}

TEST_F(AspiredVersionsManagerBuilderTest, AddSourceChainDestructionOrder) {
  // The destructor of each adapter pushes its own name into this vector, and we
  // use it to verify the destruction order.
  std::vector<string> destruct_order;
  std::function<void(const string&)> call_on_destruct =
      [&](const string& suffix) { destruct_order.push_back(suffix); };
  auto* const adapter0 =
      new FakeStoragePathSourceAdapter("adapter0", call_on_destruct);
  auto adapter1 = std::unique_ptr<FakeStoragePathSourceAdapter>(
      new FakeStoragePathSourceAdapter("adapter1", call_on_destruct));
  auto adapter2 = std::unique_ptr<FakeLoaderSourceAdapter>(
      new FakeLoaderSourceAdapter("adapter2", call_on_destruct));
  builder_->AddSourceChain(
      std::unique_ptr<FakeStoragePathSourceAdapter>(adapter0),
      std::move(adapter1), std::move(adapter2));
  std::unique_ptr<Manager> manager = builder_->Build();

  manager.reset();
  EXPECT_THAT(destruct_order, ElementsAre("adapter0", "adapter1", "adapter2"));
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
