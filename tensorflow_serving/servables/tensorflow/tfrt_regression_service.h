/* Copyright 2020 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_TFRT_REGRESSION_SERVICE_H_
#define TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_TFRT_REGRESSION_SERVICE_H_

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow_serving/apis/regression.pb.h"
#include "tensorflow_serving/model_servers/server_core.h"
#include "tensorflow_serving/servables/tensorflow/servable.h"

namespace tensorflow {
namespace serving {

// Utility methods for implementation of
// tensorflow_serving/apis/prediction_service.proto
class TFRTRegressionServiceImpl final {
 public:
  static Status Regress(const Servable::RunOptions& run_options,
                        ServerCore* core, const RegressionRequest& request,
                        RegressionResponse* response);

  // Like Regress(), but uses 'model_spec' instead of the one embedded in
  // 'request'.
  static Status RegressWithModelSpec(const Servable::RunOptions& run_options,
                                     ServerCore* core,
                                     const ModelSpec& model_spec,
                                     const RegressionRequest& request,
                                     RegressionResponse* response);
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_TFRT_REGRESSION_SERVICE_H_
