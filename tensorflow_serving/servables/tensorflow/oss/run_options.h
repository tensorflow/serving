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

#ifndef THIRD_PARTY_TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_OSS_RUN_OPTIONS_H_
#define THIRD_PARTY_TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_OSS_RUN_OPTIONS_H_

#include "tensorflow_serving/servables/tensorflow/run_options_base.h"

namespace tensorflow {
namespace serving {
namespace servables {

// RunOptions group the configuration for individual inference executions.
// The per-request configuration (e.g. deadline) can be passed here.
struct RunOptions : public RunOptionsBase {};

}  // namespace servables
}  // namespace serving
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_OSS_RUN_OPTIONS_H_
