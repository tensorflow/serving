/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow_serving/model_servers/device_runner_init_stub.h"

#include "absl/base/attributes.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/tfrt/runtime/runtime.h"

namespace tensorflow::serving {

namespace {
Status InitializeDeviceRunnerAndTopologyStub(tfrt_stub::Runtime&, int*, int*,
                                             const DeviceRunnerOptions&) {
  return tensorflow::errors::Internal(
      "device_runner_init_impl is not linked into this binary");
}
}  // namespace

Status InitializeDeviceRunnerAndTopology(tfrt_stub::Runtime& runtime,
                                         int* num_local_devices,
                                         int* cores_per_chip,
                                         const DeviceRunnerOptions& options) {
  return InitializeDeviceRunnerAndTopologyFunc(runtime, num_local_devices,
                                               cores_per_chip, options);
}

ABSL_CONST_INIT InitializeDeviceRunnerAndTopologyFuncType
    InitializeDeviceRunnerAndTopologyFunc =
        InitializeDeviceRunnerAndTopologyStub;

}  // namespace tensorflow::serving
