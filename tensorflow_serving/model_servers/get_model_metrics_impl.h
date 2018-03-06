#ifndef TENSORFLOW_SERVING_MODEL_SERVERS_GET_MODEL_METRICS_IMPL_H_
#define TENSORFLOW_SERVING_MODEL_SERVERS_GET_MODEL_METRICS_IMPL_H_

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_serving/apis/get_model_metrics.pb.h"
#include "tensorflow_serving/model_servers/server_core.h"

namespace tensorflow {
namespace serving {

class GetModelMetricsImpl {
 public:
  static Status GetModelMetrics(ServerCore* core,
                                const GetModelMetricsRequest& request,
                                GetModelMetricsResponse* response);
};

}  // namespace serving
}  // namespace tensorflow

#endif //TENSORFLOW_SERVING_MODEL_SERVERS_GET_MODEL_METRICS_IMPL_H_
