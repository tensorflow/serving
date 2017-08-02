
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
//
// Interface for performing regression using regression messages.

#ifndef TENSORFLOW_SERVING_APIS_REGRESSOR_H
#define TENSORFLOW_SERVING_APIS_REGRESSOR_H

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_serving/apis/regression.pb.h"

namespace tensorflow {
namespace serving {

class RegressorInterface {
 public:
  virtual Status Regress(const RegressionRequest& request,
                         RegressionResult* result) = 0;

  virtual ~RegressorInterface() = default;
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_APIS_REGRESSOR_H_
