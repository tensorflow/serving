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

#include "tensorflow_serving/servables/tensorflow/oss/resource_estimator.h"

#include "tensorflow/core/platform/path.h"
#include "tensorflow_serving/resources/resource_values.h"
#include "tensorflow_serving/servables/tensorflow/util.h"

namespace tensorflow {
namespace serving {

Status EstimateMainRamBytesFromValidationResult(const string& path,
                                                ResourceAllocation* estimate) {
  return errors::Unimplemented("Reading validation results is not supported");
}

Status EstimateMainRamBytesFromPath(const string& path,
                                    bool use_validation_result,
                                    FileProbingEnv* env,
                                    ResourceAllocation* estimate) {
  return EstimateResourceFromPathUsingDiskState(path, env, estimate);
}

}  // namespace serving
}  // namespace tensorflow
