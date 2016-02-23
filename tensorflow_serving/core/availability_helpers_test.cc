#include "tensorflow_serving/core/availability_helpers.h"

#include <memory>

#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
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

  std::vector<ServableId> ListAvailableServableIds() override {
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

  mutex mtx_;
  std::vector<ServableId> available_servable_ids_ GUARDED_BY(mtx_);
  int num_list_available_servable_ids_calls_ GUARDED_BY(mtx_) = 0;
  condition_variable list_available_servable_ids_condition_;
};

TEST(AvailabilityHelpersTest, Available) {
  const std::vector<ServableId> available_servables = {{"servable0", 0}};
  FakeManager fake_manager;
  fake_manager.set_available_servable_ids(available_servables);
  WaitUntilServablesAvailable(&fake_manager, available_servables);
}

TEST(AvailabilityHelpersTest, NotAvailable) {
  FakeManager fake_manager;
  fake_manager.set_available_servable_ids({{"servable0", 0}});
  Notification finished;
  std::unique_ptr<Thread> wait(
      Env::Default()->StartThread({}, "WaitUntilServablesAvailable", [&]() {
        WaitUntilServablesAvailable(&fake_manager,
                                    {{"servable0", 0}, {"servable1", 0}});
        finished.Notify();
      }));
  // Waiting for 2 calls ensures that we waited at least once for the servables
  // to be available.
  fake_manager.WaitForListAvailableServableIdsCalls(2);
  // Once this is done WaitUntilServables should stop waiting.
  fake_manager.set_available_servable_ids({{"servable0", 0}, {"servable1", 0}});
  finished.WaitForNotification();
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
