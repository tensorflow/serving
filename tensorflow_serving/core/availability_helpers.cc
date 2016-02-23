#include "tensorflow_serving/core/availability_helpers.h"

#include <algorithm>
#include <iterator>
#include <set>

#include "tensorflow_serving/util/periodic_function.h"

namespace tensorflow {
namespace serving {
namespace {

bool ServablesAvailable(Manager* const manager,
                        const std::vector<ServableId>& servables) {
  const std::set<ServableId> query_servables = {servables.begin(),
                                                servables.end()};
  const std::vector<ServableId> available_servables_list =
      manager->ListAvailableServableIds();
  const std::set<ServableId> available_servables = {
      available_servables_list.begin(), available_servables_list.end()};
  return std::includes(available_servables.begin(), available_servables.end(),
                       query_servables.begin(), query_servables.end());
}

}  // namespace

void WaitUntilServablesAvailable(Manager* const manager,
                                 const std::vector<ServableId>& servables) {
  Notification servables_available;
  PeriodicFunction periodic(
      [&]() {
        if (!servables_available.HasBeenNotified() &&
            ServablesAvailable(manager, servables)) {
          servables_available.Notify();
        }
      },
      50 * 1000 /* 50ms */);
  servables_available.WaitForNotification();
}

}  // namespace serving
}  // namespace tensorflow
