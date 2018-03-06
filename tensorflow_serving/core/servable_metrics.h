#ifndef TENSORFLOW_SERVING_CORE_SERVABLE_METRICS_H_
#define TENSORFLOW_SERVING_CORE_SERVABLE_METRICS_H_

#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace serving {

// The metrics of a servable. Typically published on an EventBus.
// These are managed/measured by ServableMetricsMonitor.
struct ServableMetrics {
  int64 request_count;
  int64 error_count;
};

// The request to update servable metrics.
struct ServableMetricsRequest {
  int64 request_value;
  int64 error_value;
};

}  // namespace serving
}  // namespace tensorflow

#endif //TENSORFLOW_SERVING_CORE_SERVABLE_METRICS_H_
