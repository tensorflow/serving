/* Copyright 2016 Google Inc. All Rights Reserved.

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

// Bootstrapping and configuration utilities for creating simple servers of
// TensorFlow models. Intended for basic instantiation with default configs.
//
// Note: All methods expect TensorFlow exports conforming to the export format
// specified at tensorflow_serving/servables/tensorflow/README.md.
#ifndef TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_SIMPLE_SERVERS_H_
#define TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_SIMPLE_SERVERS_H_

#include <memory>
#include <string>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_serving/core/dynamic_manager.h"
#include "tensorflow_serving/core/manager.h"
#include "tensorflow_serving/core/storage_path.h"
#include "tensorflow_serving/core/target.h"
#include "tensorflow_serving/session_bundle/session_bundle.h"
#include "tensorflow_serving/util/unique_ptr_with_deps.h"

namespace tensorflow {
namespace serving {
namespace simple_servers {

// TODO(b/25969594): Add full test coverage.

// Creates a Manager and associated Source for a single TensorFlow model that
// automatically loads new versions over time. All versions of the model will be
// loaded from new directories under the specified base path. Uses default
// SessionOptions.
//
// The servables loaded and served from this manager are of type
// tensorflow::serving::SessionBundle.
//
// When new versions arrive the Manager will unload the previous version before
// loading the new version. This is preferable from a resource utilization
// perspective, but has reduced availability.
Status CreateSingleTFModelManagerFromBasePath(
    const string& base_path, UniquePtrWithDeps<Manager>* manager);

}  // namespace simple_servers
}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_SIMPLE_SERVERS_H_
