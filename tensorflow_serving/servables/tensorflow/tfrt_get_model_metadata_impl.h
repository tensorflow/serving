/* Copyright 2021 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_TFRT_GET_MODEL_METADATA_IMPL_H_
#define TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_TFRT_GET_MODEL_METADATA_IMPL_H_

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_serving/apis/get_model_metadata.pb.h"
#include "tensorflow_serving/model_servers/server_core.h"

namespace tensorflow {
namespace serving {

class TFRTGetModelMetadataImpl {
 public:
  static Status GetModelMetadata(ServerCore* core,
                                 const GetModelMetadataRequest& request,
                                 GetModelMetadataResponse* response);

  // Like GetModelMetadata(), but uses 'model_spec' instead of the one embedded
  // in 'request'.
  static Status GetModelMetadataWithModelSpec(
      ServerCore* core, const ModelSpec& model_spec,
      const GetModelMetadataRequest& request,
      GetModelMetadataResponse* response);
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_TFRT_GET_MODEL_METADATA_IMPL_H_
