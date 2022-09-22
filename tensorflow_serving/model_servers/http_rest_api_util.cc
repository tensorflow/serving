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

#include "tensorflow_serving/model_servers/http_rest_api_util.h"

#include "google/protobuf/util/json_util.h"
#include "absl/strings/numbers.h"
#include <curl/curl.h>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow_serving/servables/tensorflow/get_model_metadata_impl.h"

namespace tensorflow {
namespace serving {

const char* const kPredictionApiGegex =
    R"((?i)/v1/models/([^/:]+)(?:(?:/versions/(\d+))|(?:/labels/([^/:]+)))?:(classify|regress|predict))";
const char* const kModelStatusApiRegex =
    R"((?i)/v1/models(?:/([^/:]+))?(?:(?:/versions/(\d+))|(?:/labels/([^/:]+)))?(?:\/(metadata))?)";

void AddHeaders(std::vector<std::pair<string, string>>* headers) {
  headers->push_back({"Content-Type", "application/json"});
}

void AddCORSHeaders(std::vector<std::pair<string, string>>* headers) {
  headers->push_back({"Access-Control-Allow-Origin", "*"});
  headers->push_back({"Access-Control-Allow-Methods", "POST, GET"});
  headers->push_back({"Access-Control-Allow-Headers", "Content-Type"});
}

Status FillModelSpecWithNameVersionAndLabel(
    const absl::string_view model_name,
    const absl::optional<int64_t>& model_version,
    const absl::optional<absl::string_view> model_version_label,
    ::tensorflow::serving::ModelSpec* model_spec) {
  model_spec->set_name(string(model_name));

  if (model_version.has_value() && model_version_label.has_value()) {
    return errors::InvalidArgument(
        "Both model version (", model_version.value(),
        ") and model version label (", model_version_label.value(),
        ") cannot be supplied.");
  }

  if (model_version.has_value()) {
    model_spec->mutable_version()->set_value(model_version.value());
  }
  if (model_version_label.has_value()) {
    model_spec->set_version_label(string(model_version_label.value()));
  }
  return OkStatus();
}

bool DecodeArg(string* arg) {
  static const bool run_once ABSL_ATTRIBUTE_UNUSED = [&]() {
    curl_global_init(CURL_GLOBAL_ALL);
    return true;
  }();
  CURL* curl = curl_easy_init();
  if (curl != nullptr) {
    int outlength;
    char* cres =
        curl_easy_unescape(curl, arg->c_str(), arg->size(), &outlength);
    if (cres == nullptr) {
      return false;
    }
    arg->assign(cres, outlength);
    curl_free(cres);
    curl_easy_cleanup(curl);
    return true;
  }
  return false;
}

Status ParseModelInfo(const absl::string_view http_method,
                      const absl::string_view request_path, string* model_name,
                      absl::optional<int64_t>* model_version,
                      absl::optional<string>* model_version_label,
                      string* method, string* model_subresource,
                      bool* parse_successful) {
  string model_version_str;
  string model_version_label_str;
  // Parse request parameters
  if (http_method == "POST") {
    *parse_successful =
        RE2::FullMatch(string(request_path), kPredictionApiGegex, model_name,
                       &model_version_str, &model_version_label_str, method);
  } else if (http_method == "GET") {
    *parse_successful = RE2::FullMatch(
        string(request_path), kModelStatusApiRegex, model_name,
        &model_version_str, &model_version_label_str, model_subresource);
  }
  if (!model_name->empty()) {
    if (!DecodeArg(model_name)) {
      return errors::InvalidArgument("Failed to decode model name:",
                                     *model_name);
    }
  }
  if (!model_version_str.empty()) {
    int64_t version;
    if (!absl::SimpleAtoi(model_version_str, &version)) {
      return errors::InvalidArgument(
          "Failed to convert version: ", model_version_str, " to numeric.");
    }
    *model_version = version;
  }
  if (!model_version_label_str.empty()) {
    if (!DecodeArg(&model_version_label_str)) {
      return errors::InvalidArgument("Failed to decode model version label:",
                                     model_version_label_str);
    }
    *model_version_label = model_version_label_str;
  }
  return OkStatus();
}

Status ToJsonString(const GetModelStatusResponse& response, string* output) {
  google::protobuf::util::JsonPrintOptions opts;
  opts.add_whitespace = true;
  opts.always_print_primitive_fields = true;
  // Note this is protobuf::util::Status (not TF Status) object.
  const auto& status = MessageToJsonString(response, output, opts);
  if (!status.ok()) {
    return errors::Internal("Failed to convert proto to json. Error: ",
                            status.ToString());
  }
  return OkStatus();
}

Status ToJsonString(const GetModelMetadataResponse& response, string* output) {
  google::protobuf::util::JsonPrintOptions opts;
  opts.add_whitespace = true;
  opts.always_print_primitive_fields = true;
  // TODO(b/118381513): preserving proto field names on 'Any' fields has been
  // fixed in the master branch of OSS protobuf but the TF ecosystem is
  // currently using v3.6.0 where the fix is not present. To resolve the issue
  // we invoke MessageToJsonString on invididual fields and concatenate the
  // resulting strings and make it valid JSON that conforms with the response we
  // expect.
  opts.preserve_proto_field_names = true;

  string model_spec_output;
  const auto& status1 =
      MessageToJsonString(response.model_spec(), &model_spec_output, opts);
  if (!status1.ok()) {
    return errors::Internal(
        "Failed to convert model spec proto to json. Error: ",
        status1.ToString());
  }

  tensorflow::serving::SignatureDefMap signature_def_map;
  if (response.metadata().end() ==
      response.metadata().find(GetModelMetadataImpl::kSignatureDef)) {
    return errors::Internal(
        "Failed to find 'signature_def' key in the GetModelMetadataResponse "
        "metadata map.");
  }
  bool unpack_status = response.metadata()
                           .at(GetModelMetadataImpl::kSignatureDef)
                           .UnpackTo(&signature_def_map);
  if (!unpack_status) {
    return errors::Internal(
        "Failed to unpack 'Any' object to 'SignatureDefMap'.");
  }

  string signature_def_output;
  const auto& status2 =
      MessageToJsonString(signature_def_map, &signature_def_output, opts);
  if (!status2.ok()) {
    return errors::Internal(
        "Failed to convert signature def proto to json. Error: ",
        status2.ToString());
  }

  // Concatenate the resulting strings into a valid JSON format.
  absl::StrAppend(output, "{\n");
  absl::StrAppend(output, "\"model_spec\":", model_spec_output, ",\n");
  absl::StrAppend(output, "\"metadata\": {");
  absl::StrAppend(output, "\"signature_def\": ", signature_def_output, "}\n");
  absl::StrAppend(output, "}\n");

  return OkStatus();
}

}  // namespace serving
}  // namespace tensorflow
