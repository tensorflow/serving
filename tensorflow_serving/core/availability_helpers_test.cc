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

#include "tensorflow_serving/core/availability_helpers.h"

#include <memory>

#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow_serving/core/manager.h"
#include "tensorflow_serving/core/servable_handle.h"

namespace tensorflow {
namespace serving {
namespace {

class FakeManager : public Manager {
 public:
  FakeManager() {}

  void set_available_servable_ids(
      const std::vector<ServableId>& available_servable_ids) {
    mutex_lock l(mtx_);
    available_servable_ids_ = available_servable_ids;
  }

  std::vector<ServableId> ListAvailableServableIds() const override {
    mutex_lock l(mtx_);
    ++num_list_available_servable_ids_calls_;
    list_available_servable_ids_condition_.notify_all();
    return available_servable_ids_;
  }

  void WaitForListAvailableServableIdsCalls(const int n) {
    mutex_lock l(mtx_);
    while (n < num_list_available_servable_ids_calls_) {
      list_available_servable_ids_condition_.wait(l);
    }
  }

 private:
  Status GetUntypedServableHandle(
      const ServableRequest& request,
      std::unique_ptr<UntypedServableHandle>* result) override {
    LOG(FATAL) << "Not expected to be called.";
  }

  std::map<ServableId, std::unique_ptr<UntypedServableHandle>>
  GetAvailableUntypedServableHandles() const override {
    LOG(FATAL) << "Not expected to be called.";
  }

  mutable mutex mtx_;
  std::vector<ServableId> available_servable_ids_ GUARDED_BY(mtx_);
  mutable int num_list_available_servable_ids_calls_ GUARDED_BY(mtx_) = 0;
  mutable condition_variable list_available_servable_ids_condition_;
};

TEST(AvailabilityHelpersTest, SpecificAvailable) {
  FakeManager fake_manager;
  fake_manager.set_available_servable_ids({{"servable0", 0}});
  WaitUntilServablesAvailable({{"servable0", 0}}, &fake_manager);
}

TEST(AvailabilityHelpersTest, SpecificNotAvailable) {
  FakeManager fake_manager;
  fake_manager.set_available_servable_ids({{"servable0", 0}});
  Notification finished;
  std::unique_ptr<Thread> wait(
      Env::Default()->StartThread({}, "WaitUntilServablesAvailable", [&]() {
        WaitUntilServablesAvailable({{"servable0", 0}, {"servable1", 0}},
                                    &fake_manager);
        finished.Notify();
      }));
  // Waiting for 2 calls ensures that we waited at least once for the servables
  // to be available.
  fake_manager.WaitForListAvailableServableIdsCalls(2);
  // Once this is done WaitUntilServables should stop waiting.
  fake_manager.set_available_servable_ids({{"servable0", 0}, {"servable1", 0}});
  finished.WaitForNotification();
}

TEST(AvailabilityHelpersTest, LatestNotAvailable) {
  FakeManager fake_manager;
  fake_manager.set_available_servable_ids({{"servable0", 0}});
  Notification finished;
  std::unique_ptr<Thread> wait(
      Env::Default()->StartThread({}, "WaitUntilServablesAvailable", [&]() {
        WaitUntilServablesAvailableForRequests(
            {ServableRequest::Specific("servable0", 0),
             ServableRequest::Latest("servable1")},
            &fake_manager);
        finished.Notify();
      }));
  // Waiting for 2 calls ensures that we waited at least once for the servables
  // to be available.
  fake_manager.WaitForListAvailableServableIdsCalls(2);
  // Once this is done WaitUntilServables should stop waiting.
  fake_manager.set_available_servable_ids({{"servable0", 0}, {"servable1", 2}});
  finished.WaitForNotification();
}

TEST(AvailabilityHelpersTest, LatestVersion) {
  FakeManager fake_manager;
  fake_manager.set_available_servable_ids({{"servable0", 123}});
  WaitUntilServablesAvailableForRequests({ServableRequest::Latest("servable0")},
                                         &fake_manager);
}

TEST(AvailabilityHelpersTest, LatestAndExactVersion) {
  FakeManager fake_manager;
  fake_manager.set_available_servable_ids({{"servable0", 0}, {"servable1", 1}});
  WaitUntilServablesAvailableForRequests(
      {ServableRequest::Latest("servable0"),
       ServableRequest::Specific("servable1", 1)},
      &fake_manager);
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
