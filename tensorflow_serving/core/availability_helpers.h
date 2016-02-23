// Helper methods related to the availability of servables.

#ifndef TENSORFLOW_SERVING_CORE_AVAILABILITY_HELPERS_H_
#define TENSORFLOW_SERVING_CORE_AVAILABILITY_HELPERS_H_

#include <vector>

#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow_serving/core/manager.h"
#include "tensorflow_serving/core/servable_id.h"

namespace tensorflow {
namespace serving {

// Waits until all of these servables are available for serving through the
// provided manager.
//
// Implementation notes:
// The method is implemented by polling the manager, and the periodicity of
// polling is 50ms.  We call ListAvailableServableIds on the manager, which may
// have side-effects for certain manager implementations, e.g. causing servables
// to be loaded.
void WaitUntilServablesAvailable(Manager* manager,
                                 const std::vector<ServableId>& servables);

// Implementation details follow. Please do not depend on these!
// We expose these for test accessibility.
namespace internal {
// Returns true if all of these servable ids are available through the provided
// manager.
//
// Implementation notes:
// We call ListAvailableServableIds on the manager, which may have side-effects
// for certain manager implementations, e.g. causing servables to be loaded.
bool ServablesAvailable(Manager* manager,
                        const std::vector<ServableId>& servables);
}  // namespace internal.

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_AVAILABILITY_HELPERS_H_
