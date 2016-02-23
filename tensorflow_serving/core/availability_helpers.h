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
// Servables can be specified in two ways:
//   1. As specific versions, in which case one or more specific versions can be
//      specified for a servable stream name, and must match exactly.
//   2. As latest versions, in which case any version available for a servable
//      stream name will be matched.
//
// Implementation notes:
// The method is implemented by polling the manager, and the periodicity of
// polling is 50ms.  We call ListAvailableServableIds on the manager, which may
// have side-effects for certain manager implementations, e.g. causing servables
// to be loaded.
void WaitUntilServablesAvailable(Manager* manager,
                                 const std::vector<ServableRequest>& servables);

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_AVAILABILITY_HELPERS_H_
