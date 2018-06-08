/* Copyright 2018 Google Inc. All Rights Reserved.

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
#ifndef TENSORFLOW_SERVING_MODEL_SERVERS_HTTP_REST_PREDICTION_HANDLER_H_
#define TENSORFLOW_SERVING_MODEL_SERVERS_HTTP_REST_PREDICTION_HANDLER_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "re2/re2.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace tensorflow {

class SignatureDef;

namespace serving {

class ServerCore;
class TensorflowPredictor;
class ModelSpec;

// HttpRestPredictionHandler handles HTTP/REST APIs of TF serving.
//
// Currently supported APIs are as follows:
//
// o Predict
//
//   Paths:
//   /v1/models/<model_name>:predict (uses 'latest' version of the model).
//   /v1/models/<model_name>/versions/<version_number>:predict
//
//   Request/Response format:
//   https://cloud.google.com/ml-engine/docs/v1/predict-request
//
//
// Users of this class should typically create one instance of it at process
// startup, register paths defined by kPathRegex with the in-process HTTP
// server, and when a request arrives, forward the request to ProcessRequest()
// method.
//
// This class is thread safe.
class HttpRestPredictionHandler {
 public:
  // Returns a regex that captures all API paths handled by this handler.
  // Typical use of this method is to register request paths with underlying
  // HTTP server, so incoming requests can be forwarded to this handler.
  static const char* const kPathRegex;

  // API calls are configured to timeout after `run_optons.timeout_in_ms`.
  // `core` is not owned and is expected to outlive HttpRestPredictionHandler
  // instance.
  HttpRestPredictionHandler(const RunOptions& run_options, ServerCore* core);

  ~HttpRestPredictionHandler();

  // Process a HTTP request.
  //
  // If `http_method` (e.g. POST) and `request_path` (e.g. /v1/models/m:predict)
  // match one of the supported APIs, the body (JSON object) is processed and
  // response (JSON object) is returned in `output` along with output `headers`.
  //
  // In case of errors, the `headers` and `output` are still relevant as they
  // contain detailed error messages, that can be relayed back to the client.
  Status ProcessRequest(const absl::string_view http_method,
                        const absl::string_view request_path,
                        const absl::string_view request_body,
                        std::vector<std::pair<string, string>>* headers,
                        string* output);

 private:
  Status ProcessClassifyRequest(const absl::string_view model_name,
                                const absl::optional<int64>& model_version,
                                const absl::string_view request_body,
                                string* output);
  Status ProcessRegressRequest(const absl::string_view model_name,
                               const absl::optional<int64>& model_version,
                               const absl::string_view request_body,
                               string* output);
  Status ProcessPredictRequest(const absl::string_view model_name,
                               const absl::optional<int64>& model_version,
                               const absl::string_view request_body,
                               string* output);

  Status GetInfoMap(const ModelSpec& model_spec, const string& signature_name,
                    ::google::protobuf::Map<string, tensorflow::TensorInfo>* infomap);

  const RunOptions run_options_;
  ServerCore* core_;
  std::unique_ptr<TensorflowPredictor> predictor_;
  const RE2 prediction_api_regex_;
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_MODEL_SERVERS_HTTP_REST_PREDICTION_HANDLER_H_
