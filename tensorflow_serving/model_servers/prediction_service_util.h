/* Copyright 2022 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_SERVING_MODEL_SERVERS_PREDICTION_SERVICE_UTIL_H_
#define TENSORFLOW_SERVING_MODEL_SERVERS_PREDICTION_SERVICE_UTIL_H_

#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#include "tensorflow_serving/model_servers/server_core.h"
#include "tensorflow_serving/servables/tensorflow/thread_pool_factory.h"

namespace tensorflow {
namespace serving {

// Options for configuring a PredictionServiceBase object.
struct PredictionServiceOptions {
  ServerCore* server_core;
  bool enforce_session_run_timeout;
  ThreadPoolFactory* thread_pool_factory = nullptr;
};

// Convert the request deadline represented in absolute time point into number
// of milliseconds from now.
int DeadlineToTimeoutMillis(const gpr_timespec deadline);

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_MODEL_SERVERS_PREDICTION_SERVICE_UTIL_H_
