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

#include "tensorflow_serving/core/router.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow_serving/core/servable_data.h"
#include "tensorflow_serving/core/storage_path.h"
#include "tensorflow_serving/core/test_util/mock_storage_path_target.h"

using ::testing::_;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::IsEmpty;
using ::testing::Return;
using ::testing::StrictMock;

namespace tensorflow {
namespace serving {
namespace {

class TestRouter : public Router<StoragePath> {
 public:
  TestRouter() = default;
  ~TestRouter() override = default;

 protected:
  int num_output_ports() const override { return 2; }

  int Route(const StringPiece servable_name,
            const std::vector<ServableData<StoragePath>>& versions) override {
    if (servable_name == "zero") {
      return 0;
    } else if (servable_name == "one") {
      return 1;
    } else {
      LOG(FATAL) << "Unexpected test data";
    }
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(TestRouter);
};

TEST(RouterTest, Basic) {
  TestRouter router;
  std::vector<Source<StoragePath>*> output_ports = router.GetOutputPorts();
  ASSERT_EQ(2, output_ports.size());
  std::vector<std::unique_ptr<test_util::MockStoragePathTarget>> targets;
  for (int i = 0; i < output_ports.size(); ++i) {
    std::unique_ptr<test_util::MockStoragePathTarget> target(
        new StrictMock<test_util::MockStoragePathTarget>);
    ConnectSourceToTarget(output_ports[i], target.get());
    targets.push_back(std::move(target));
  }

  EXPECT_CALL(*targets[0],
              SetAspiredVersions(
                  Eq("zero"),
                  ElementsAre(ServableData<StoragePath>({"zero", 0}, "mrop"))));
  router.SetAspiredVersions("zero",
                            {ServableData<StoragePath>({"zero", 0}, "mrop")});

  EXPECT_CALL(*targets[1], SetAspiredVersions(
                               Eq("one"), ElementsAre(ServableData<StoragePath>(
                                              {"one", 1}, "floo"))));
  router.SetAspiredVersions("one",
                            {ServableData<StoragePath>({"one", 1}, "floo")});
}

TEST(RouterTest, SetAspiredVersionsBlocksUntilAllTargetsConnected_1) {
  // Scenario 1: When SetAspiredVersions() is invoked, GetOutputPorts() has not
  // yet been called. The SetAspiredVersions() call should block until the ports
  // have been emitted and all of them have been connected to targets.

  TestRouter router;
  Notification done;

  // Connect the output ports to targets asynchronously, after a long delay.
  std::unique_ptr<Thread> connect_targets(Env::Default()->StartThread(
      {}, "ConnectTargets",
      [&router, &done] {
        // Sleep for a long time before calling GetOutputPorts(), to make it
        // very likely that SetAspiredVersions() gets called first and has to
        // block.
        Env::Default()->SleepForMicroseconds(1 * 1000 * 1000 /* 1 second */);

        std::vector<Source<StoragePath>*> output_ports =
            router.GetOutputPorts();
        ASSERT_EQ(2, output_ports.size());
        std::vector<std::unique_ptr<test_util::MockStoragePathTarget>> targets;
        for (int i = 0; i < output_ports.size(); ++i) {
          std::unique_ptr<test_util::MockStoragePathTarget> target(
              new StrictMock<test_util::MockStoragePathTarget>);
          EXPECT_CALL(*target, SetAspiredVersions(_, IsEmpty()));
          ConnectSourceToTarget(output_ports[i], target.get());
          targets.push_back(std::move(target));
        }
        done.WaitForNotification();
      }));

  router.SetAspiredVersions("zero", {});
  router.SetAspiredVersions("one", {});

  done.Notify();
}

TEST(RouterTest, SetAspiredVersionsBlocksUntilAllTargetsConnected_2) {
  // Scenario 2: When SetAspiredVersions() is invoked, GetOutputPorts() has been
  // called but only one of the two ports has been connected to a target. The
  // SetAspiredVersions() call should block until the other port is connected.

  TestRouter router;
  std::vector<Source<StoragePath>*> output_ports = router.GetOutputPorts();
  ASSERT_EQ(2, output_ports.size());
  std::vector<std::unique_ptr<test_util::MockStoragePathTarget>> targets;
  for (int i = 0; i < output_ports.size(); ++i) {
    std::unique_ptr<test_util::MockStoragePathTarget> target(
        new StrictMock<test_util::MockStoragePathTarget>);
    targets.push_back(std::move(target));
  }

  // Connect target 0 now.
  ConnectSourceToTarget(output_ports[0], targets[0].get());

  // Connect target 1 asynchronously after a long delay.
  std::unique_ptr<Thread> connect_target_1(Env::Default()->StartThread(
      {}, "ConnectTarget1",
      [&output_ports, &targets] {
        // Sleep for a long time before connecting target 1, to make it very
        // likely that SetAspiredVersions() gets called first and has to
        // block.
        Env::Default()->SleepForMicroseconds(1 * 1000 * 1000 /* 1 second */);
        ConnectSourceToTarget(output_ports[1], targets[1].get());
      }));

  EXPECT_CALL(*targets[1], SetAspiredVersions(Eq("one"), IsEmpty()));
  router.SetAspiredVersions("one", {});
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
