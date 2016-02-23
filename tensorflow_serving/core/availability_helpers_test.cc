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
    return available_servable_ids_;
  }

 private:
  Status GetUntypedServableHandle(
      const ServableRequest& request,
      std::unique_ptr<UntypedServableHandle>* result) override {
    LOG(FATAL) << "Not expected to be called.";
  }

  mutex mtx_;
  std::vector<ServableId> available_servable_ids_ GUARDED_BY(mtx_);
};

TEST(AvailabilityHelpersTest, Available) {
  const std::vector<ServableId> available_servables = {{"servable0", 0}};
  FakeManager fake_manager;
  fake_manager.set_available_servable_ids(available_servables);
  WaitUntilServablesAvailable(&fake_manager, available_servables);
}

TEST(AvailabilityHelpersTest, NotAvailableNonWaiting) {
  FakeManager fake_manager;
  fake_manager.set_available_servable_ids({{"servable0", 0}});
  ASSERT_FALSE(internal::ServablesAvailable(
      &fake_manager, {{"servable0", 0}, {"servable1", 0}}));
}

TEST(AvailabilityHelpersTest, NotAvailableWaiting) {
  FakeManager fake_manager;
  fake_manager.set_available_servable_ids({{"servable0", 0}});
  Notification started;
  Notification finished;
  std::unique_ptr<Thread> wait(
      Env::Default()->StartThread({}, "WaitUntilServablesAvailable", [&]() {
        started.Notify();
        WaitUntilServablesAvailable(&fake_manager,
                                    {{"servable0", 0}, {"servable1", 0}});
        finished.Notify();
      }));
  started.WaitForNotification();
  // Once this is done WaitUntilServables should stop waiting.
  fake_manager.set_available_servable_ids({{"servable0", 0}, {"servable1", 0}});
  finished.WaitForNotification();
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
