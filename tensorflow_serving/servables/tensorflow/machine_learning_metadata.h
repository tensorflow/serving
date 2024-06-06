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

#ifndef TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_MACHINE_LEARNING_METADATA_H_
#define TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_MACHINE_LEARNING_METADATA_H_

#include <string>

#include "tensorflow/core/platform/status.h"

namespace tensorflow {
namespace serving {

// If present, processes Machine Learning Metadata associated with the
// SavedModel. Currently, this broadcasts the MLMD UUID as a key associated
// with a loaded model.
// For more information: https://www.tensorflow.org/tfx/guide/mlmd
void MaybePublishMLMDStreamz(const string& export_dir, const string& model_name,
                             int64_t version);

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_MACHINE_LEARNING_METADATA_H_
