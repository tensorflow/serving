/* Copyright 2018 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_SERVING_MODEL_SERVERS_GET_MODEL_STATUS_IMPL_H_
#define TENSORFLOW_SERVING_MODEL_SERVERS_GET_MODEL_STATUS_IMPL_H_

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_serving/apis/get_model_status.pb.h"
#include "tensorflow_serving/model_servers/server_core.h"

namespace tensorflow {
namespace serving {

// Returns response with status information for model. If the request
// specifies a model version, information about only that version will be
// returned. If no version is specified, information about all versions of the
// model will be returned.
class GetModelStatusImpl {
 public:
  static Status GetModelStatus(ServerCore* core,
                               const GetModelStatusRequest& request,
                               GetModelStatusResponse* response);

  // Like GetModelStatus(), but uses 'model_spec' instead of the one embedded in
  // 'request'.
  static Status GetModelStatusWithModelSpec(
      ServerCore* core, const ModelSpec& model_spec,
      const GetModelStatusRequest& request, GetModelStatusResponse* response);
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_MODEL_SERVERS_GET_MODEL_STATUS_IMPL_H_
