#include "tensorflow_serving/core/availability_helpers.h"

#include <algorithm>
#include <iterator>
#include <set>

#include "tensorflow/core/platform/env.h"

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
  while (!ServablesAvailable(manager, servables)) {
    Env::Default()->SleepForMicroseconds(50 * 1000 /* 50 ms */);
  }
}

}  // namespace serving
}  // namespace tensorflow
