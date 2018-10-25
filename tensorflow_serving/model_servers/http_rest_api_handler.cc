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

#include "tensorflow_serving/model_servers/http_rest_api_handler.h"

#include <string>

#include "google/protobuf/any.pb.h"
#include "google/protobuf/util/json_util.h"
#include "absl/strings/escaping.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/time/time.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow_serving/apis/model.pb.h"
#include "tensorflow_serving/apis/predict.pb.h"
#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/model_servers/get_model_status_impl.h"
#include "tensorflow_serving/model_servers/server_core.h"
#include "tensorflow_serving/servables/tensorflow/classification_service.h"
#include "tensorflow_serving/servables/tensorflow/get_model_metadata_impl.h"
#include "tensorflow_serving/servables/tensorflow/predict_impl.h"
#include "tensorflow_serving/servables/tensorflow/regression_service.h"
#include "tensorflow_serving/util/json_tensor.h"

namespace tensorflow {
namespace serving {

using protobuf::util::JsonPrintOptions;
using protobuf::util::MessageToJsonString;
using tensorflow::serving::ServerCore;
using tensorflow::serving::TensorflowPredictor;

const char* const HttpRestApiHandler::kPathRegex = "(?i)/v1/.*";

HttpRestApiHandler::HttpRestApiHandler(const RunOptions& run_options,
                                       ServerCore* core)
    : run_options_(run_options),
      core_(core),
      predictor_(new TensorflowPredictor(true /* use_saved_model */)),
      prediction_api_regex_(
          R"((?i)/v1/models/([^/:]+)(?:/versions/(\d+))?:(classify|regress|predict))"),
      modelstatus_api_regex_(
          R"((?i)/v1/models(?:/([^/:]+))?(?:/versions/(\d+))?(?:\/(metadata))?)") {
}

HttpRestApiHandler::~HttpRestApiHandler() {}

namespace {

void AddHeaders(std::vector<std::pair<string, string>>* headers) {
  headers->push_back({"Content-Type", "application/json"});
}

void FillJsonErrorMsg(const string& errmsg, string* output) {
  // Errors are represented as following JSON object:
  // {
  //   "error": "<CEscaped error message string>"
  // }
  absl::StrAppend(output, R"({ "error": ")", absl::CEscape(errmsg), R"(" })");
}

}  // namespace

Status HttpRestApiHandler::ProcessRequest(
    const absl::string_view http_method, const absl::string_view request_path,
    const absl::string_view request_body,
    std::vector<std::pair<string, string>>* headers, string* output) {
  headers->clear();
  output->clear();
  AddHeaders(headers);
  string model_name;
  string model_version_str;
  string method;
  string model_subresource;
  Status status = errors::InvalidArgument("Malformed request: ", http_method,
                                          " ", request_path);
  if (http_method == "POST" &&
      RE2::FullMatch(string(request_path), prediction_api_regex_, &model_name,
                     &model_version_str, &method)) {
    absl::optional<int64> model_version;
    if (!model_version_str.empty()) {
      int64 version;
      if (!absl::SimpleAtoi(model_version_str, &version)) {
        return errors::InvalidArgument(
            "Failed to convert version: ", model_version_str, " to numeric.");
      }
      model_version = version;
    }
    if (method == "classify") {
      status = ProcessClassifyRequest(model_name, model_version, request_body,
                                      output);
    } else if (method == "regress") {
      status = ProcessRegressRequest(model_name, model_version, request_body,
                                     output);
    } else if (method == "predict") {
      status = ProcessPredictRequest(model_name, model_version, request_body,
                                     output);
    }
  } else if (http_method == "GET" &&
             RE2::FullMatch(string(request_path), modelstatus_api_regex_,
                            &model_name, &model_version_str,
                            &model_subresource)) {
    if (!model_subresource.empty() && model_subresource == "metadata") {
      status =
          ProcessModelMetadataRequest(model_name, model_version_str, output);
    } else {
      status = ProcessModelStatusRequest(model_name, model_version_str, output);
    }
  }

  if (!status.ok()) {
    FillJsonErrorMsg(status.error_message(), output);
  }
  return status;
}

Status HttpRestApiHandler::ProcessClassifyRequest(
    const absl::string_view model_name,
    const absl::optional<int64>& model_version,
    const absl::string_view request_body, string* output) {
  ClassificationRequest request;
  request.mutable_model_spec()->set_name(string(model_name));
  if (model_version.has_value()) {
    request.mutable_model_spec()->mutable_version()->set_value(
        model_version.value());
  }
  TF_RETURN_IF_ERROR(FillClassificationRequestFromJson(request_body, &request));

  ClassificationResponse response;
  TF_RETURN_IF_ERROR(TensorflowClassificationServiceImpl::Classify(
      run_options_, core_, request, &response));
  TF_RETURN_IF_ERROR(
      MakeJsonFromClassificationResult(response.result(), output));
  return Status::OK();
}

Status HttpRestApiHandler::ProcessRegressRequest(
    const absl::string_view model_name,
    const absl::optional<int64>& model_version,
    const absl::string_view request_body, string* output) {
  RegressionRequest request;
  request.mutable_model_spec()->set_name(string(model_name));
  if (model_version.has_value()) {
    request.mutable_model_spec()->mutable_version()->set_value(
        model_version.value());
  }
  TF_RETURN_IF_ERROR(FillRegressionRequestFromJson(request_body, &request));

  RegressionResponse response;
  TF_RETURN_IF_ERROR(TensorflowRegressionServiceImpl::Regress(
      run_options_, core_, request, &response));
  TF_RETURN_IF_ERROR(MakeJsonFromRegressionResult(response.result(), output));
  return Status::OK();
}

Status HttpRestApiHandler::ProcessPredictRequest(
    const absl::string_view model_name,
    const absl::optional<int64>& model_version,
    const absl::string_view request_body, string* output) {
  PredictRequest request;
  request.mutable_model_spec()->set_name(string(model_name));
  if (model_version.has_value()) {
    request.mutable_model_spec()->mutable_version()->set_value(
        model_version.value());
  }
  JsonPredictRequestFormat format;
  TF_RETURN_IF_ERROR(FillPredictRequestFromJson(
      request_body,
      [this, &request](const string& sig,
                       ::google::protobuf::Map<string, TensorInfo>* map) {
        return this->GetInfoMap(request.model_spec(), sig, map);
      },
      &request, &format));

  PredictResponse response;
  TF_RETURN_IF_ERROR(
      predictor_->Predict(run_options_, core_, request, &response));
  TF_RETURN_IF_ERROR(MakeJsonFromTensors(response.outputs(), format, output));
  return Status::OK();
}

Status HttpRestApiHandler::ProcessModelStatusRequest(
    const absl::string_view model_name,
    const absl::string_view model_version_str, string* output) {
  GetModelStatusRequest request;
  // We do not yet support returning status of all models
  // to be in-sync with the gRPC GetModelStatus API.
  if (model_name.empty()) {
    return errors::InvalidArgument("Missing model name in request.");
  }
  request.mutable_model_spec()->set_name(string(model_name));
  if (!model_version_str.empty()) {
    int64 version;
    if (!absl::SimpleAtoi(model_version_str, &version)) {
      return errors::InvalidArgument(
          "Failed to convert version: ", model_version_str, " to numeric.");
    }
    request.mutable_model_spec()->mutable_version()->set_value(version);
  }

  GetModelStatusResponse response;
  TF_RETURN_IF_ERROR(
      GetModelStatusImpl::GetModelStatus(core_, request, &response));
  JsonPrintOptions opts;
  opts.add_whitespace = true;
  opts.always_print_primitive_fields = true;
  // Note this is protobuf::util::Status (not TF Status) object.
  const auto& status = MessageToJsonString(response, output, opts);
  if (!status.ok()) {
    return errors::Internal("Failed to convert proto to json. Error: ",
                            status.ToString());
  }
  return Status::OK();
}

Status HttpRestApiHandler::ProcessModelMetadataRequest(
    const absl::string_view model_name,
    const absl::string_view model_version_str, string* output) {
  GetModelMetadataRequest request;
  if (model_name.empty()) {
    return errors::InvalidArgument("Missing model name in request.");
  }
  request.mutable_model_spec()->set_name(string(model_name));
  // We currently only support the kSignatureDef metadata field
  request.add_metadata_field(GetModelMetadataImpl::kSignatureDef);
  if (!model_version_str.empty()) {
    int64 version;
    if (!absl::SimpleAtoi(model_version_str, &version)) {
      return errors::InvalidArgument(
          "Failed to convert version: ", model_version_str, " to numeric.");
    }
    request.mutable_model_spec()->mutable_version()->set_value(version);
  }

  GetModelMetadataResponse response;
  TF_RETURN_IF_ERROR(
      GetModelMetadataImpl::GetModelMetadata(core_, request, &response));
  JsonPrintOptions opts;
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
  return Status::OK();
}

Status HttpRestApiHandler::GetInfoMap(
    const ModelSpec& model_spec, const string& signature_name,
    ::google::protobuf::Map<string, tensorflow::TensorInfo>* infomap) {
  ServableHandle<SavedModelBundle> bundle;
  TF_RETURN_IF_ERROR(core_->GetServableHandle(model_spec, &bundle));
  const string& signame =
      signature_name.empty() ? kDefaultServingSignatureDefKey : signature_name;
  auto iter = bundle->meta_graph_def.signature_def().find(signame);
  if (iter == bundle->meta_graph_def.signature_def().end()) {
    return errors::InvalidArgument("Serving signature name: \"", signame,
                                   "\" not found in signature def");
  }
  *infomap = iter->second.inputs();
  return Status::OK();
}

}  // namespace serving
}  // namespace tensorflow
