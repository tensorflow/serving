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

#ifndef THIRD_PARTY_TENSORFLOW_SERVING_MODEL_SERVERS_HTTP_REST_API_UTIL_H_
#define THIRD_PARTY_TENSORFLOW_SERVING_MODEL_SERVERS_HTTP_REST_API_UTIL_H_

#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "re2/re2.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_serving/apis/get_model_metadata.pb.h"
#include "tensorflow_serving/apis/get_model_status.pb.h"
#include "tensorflow_serving/apis/model.pb.h"

namespace tensorflow {
namespace serving {

const char* const kHTTPRestApiHandlerPathRegex = "(?i)/v1/.*";

void AddHeaders(std::vector<std::pair<string, string>>* headers);

void AddCORSHeaders(std::vector<std::pair<string, string>>* headers);

Status FillModelSpecWithNameVersionAndLabel(
    const absl::string_view model_name,
    const absl::optional<int64_t>& model_version,
    const absl::optional<absl::string_view> model_version_label,
    ::tensorflow::serving::ModelSpec* model_spec);

// Parse model information from the request.
Status ParseModelInfo(const absl::string_view http_method,
                      const absl::string_view request_path, string* model_name,
                      absl::optional<int64_t>* model_version,
                      absl::optional<string>* model_version_label,
                      string* method, string* model_subresource,
                      bool* parse_successful);

Status ToJsonString(const GetModelStatusResponse& response, string* output);

Status ToJsonString(const GetModelMetadataResponse& response, string* output);

}  // namespace serving
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_SERVING_MODEL_SERVERS_HTTP_REST_API_UTIL_H_
