/* Copyright 2022 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_SERVING_MODEL_SERVERS_HTTP_REST_API_HANDLER_BASE_H_
#define TENSORFLOW_SERVING_MODEL_SERVERS_HTTP_REST_API_HANDLER_BASE_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace serving {

// Base class of HttpRestApiHandler classes that handles HTTP/REST APIs of TF
// serving.
//
// Currently supported APIs are as follows:
//
// o Inference - Classify/Regress/Predict
//
//   POST /v1/models/<model_name>:(classify|regress)
//   POST /v1/models/<model_name>/versions/<ver>:(classify|regress)
//
// o Model status
//
//   GET /v1/models/<model_name> (status of all versions)
//   GET /v1/models/<model_name>/versions/<ver> (status of specific version)
//
// The API is documented here:
// tensorflow_serving/g3doc/api_rest.md
//
// Users of this class should typically create one instance of it at process
// startup, register paths defined by kPathRegex with the in-process HTTP
// server, and when a request arrives, forward the request to ProcessRequest()
// method.
//
// This class is thread safe.
class HttpRestApiHandlerBase {
 public:
  virtual ~HttpRestApiHandlerBase() = default;

  // Process a HTTP request.
  //
  // In case of errors, the `headers` and `output` are still relevant as they
  // contain detailed error messages, that can be relayed back to the client.
  virtual Status ProcessRequest(const absl::string_view http_method,
                                const absl::string_view request_path,
                                const absl::string_view request_body,
                                std::vector<std::pair<string, string>>* headers,
                                string* model_name, string* method,
                                string* output) = 0;
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_MODEL_SERVERS_HTTP_REST_API_HANDLER_BASE_H_
