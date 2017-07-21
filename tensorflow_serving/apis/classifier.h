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

#ifndef TENSORFLOW_SERVING_APIS_CLASSIFIER_H_
#define TENSORFLOW_SERVING_APIS_CLASSIFIER_H_

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_serving/apis/classification.pb.h"

namespace tensorflow {
namespace serving {

/// Model-type agnostic interface for performing classification.
///
/// Specific implementations will exist for different model types
/// (e.g. TensorFlow SavedModel) that can convert the request into a model
/// specific input and know how to convert the output into a generic
/// ClassificationResult.
class ClassifierInterface {
 public:
  /// Given a ClassificationRequest, populates the ClassificationResult with the
  /// result.
  ///
  /// @param request  Input request specifying the model/signature to query
  /// along with the data payload.
  /// @param result   The output classifications that will get populated.
  /// @return         A status object indicating success or failure.
  virtual Status Classify(const ClassificationRequest& request,
                          ClassificationResult* result) = 0;

  virtual ~ClassifierInterface() = default;
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_APIS_CLASSIFIER_H_
