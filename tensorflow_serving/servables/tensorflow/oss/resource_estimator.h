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

#ifndef THIRD_PARTY_TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_GOOGLE_RESOURCE_ESTIMATER_H_
#define THIRD_PARTY_TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_GOOGLE_RESOURCE_ESTIMATER_H_

#include "tensorflow_serving/resources/resources.pb.h"
#include "tensorflow_serving/util/file_probing_env.h"

namespace tensorflow {
namespace serving {

// Estimates the ram resources a session bundle or saved model bundle will use
// once loaded, from its export or saved model path.
// TODO(b/150736159): Support use_validation_result. Right now,
// use_validation_result is ignored and the function always get resource
// estimation from disk.
Status EstimateMainRamBytesFromPath(const string& path,
                                    bool use_validation_result,
                                    FileProbingEnv* env,
                                    ResourceAllocation* estimate);

}  // namespace serving
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_GOOGLE_RESOURCE_ESTIMATER_H_
