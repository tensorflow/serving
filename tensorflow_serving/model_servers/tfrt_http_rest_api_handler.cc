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

#include "tensorflow_serving/model_servers/tfrt_http_rest_api_handler.h"

#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/any.pb.h"
#include "absl/status/status.h"
#include "absl/strings/escaping.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "xla/tsl/platform/errors.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/tfrt/saved_model/saved_model.h"
#include "tensorflow_serving/apis/model.pb.h"
#include "tensorflow_serving/apis/predict.pb.h"
#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/model_servers/get_model_status_impl.h"
#include "tensorflow_serving/model_servers/http_rest_api_util.h"
#include "tensorflow_serving/model_servers/server_core.h"
#include "tensorflow_serving/servables/tensorflow/servable.h"
#include "tensorflow_serving/servables/tensorflow/tfrt_get_model_metadata_impl.h"
#include "tensorflow_serving/servables/tensorflow/tfrt_servable.h"
#include "tensorflow_serving/util/json_tensor.h"

namespace tensorflow {
namespace serving {

using tensorflow::serving::ServerCore;

const char* const TFRTHttpRestApiHandler::kPathRegex =
    kHTTPRestApiHandlerPathRegex;

TFRTHttpRestApiHandler::TFRTHttpRestApiHandler(int timeout_in_ms,
                                               ServerCore* core)
    : run_options_(),
      timeout_(absl::Milliseconds(timeout_in_ms)),
      core_(core) {}

TFRTHttpRestApiHandler::~TFRTHttpRestApiHandler() {}

absl::Status TFRTHttpRestApiHandler::ProcessRequest(
    const absl::string_view http_method, const absl::string_view request_path,
    const absl::string_view request_body,
    std::vector<std::pair<std::string, std::string>>* headers,
    std::string* model_name, std::string* method, std::string* output) {
  headers->clear();
  output->clear();
  AddHeaders(headers);

  std::string model_subresource;
  absl::Status status = errors::InvalidArgument(
      "Malformed request: ", http_method, " ", request_path);
  absl::optional<int64_t> model_version;
  absl::optional<std::string> model_version_label;
  bool parse_successful;

  TF_RETURN_IF_ERROR(ParseModelInfo(
      http_method, request_path, model_name, &model_version,
      &model_version_label, method, &model_subresource, &parse_successful));

  auto run_options = run_options_;
  run_options.deadline = absl::Now() + timeout_;

  // Dispatch request to appropriate processor
  if (http_method == "POST" && parse_successful) {
    if (*method == "classify") {
      status = ProcessClassifyRequest(*model_name, model_version,
                                      model_version_label, request_body,
                                      run_options, output);
    } else if (*method == "regress") {
      status =
          ProcessRegressRequest(*model_name, model_version, model_version_label,
                                request_body, run_options, output);
    } else if (*method == "predict") {
      status =
          ProcessPredictRequest(*model_name, model_version, model_version_label,
                                request_body, run_options, output);
    }
  } else if (http_method == "GET" && parse_successful) {
    if (!model_subresource.empty() && model_subresource == "metadata") {
      status = ProcessModelMetadataRequest(*model_name, model_version,
                                           model_version_label, output);
    } else {
      status = ProcessModelStatusRequest(
          *model_name, model_version, model_version_label, run_options, output);
    }
  }

  MakeJsonFromStatus(status, output);
  return status;
}

absl::Status TFRTHttpRestApiHandler::ProcessClassifyRequest(
    const absl::string_view model_name,
    const absl::optional<int64_t>& model_version,
    const absl::optional<absl::string_view>& model_version_label,
    const absl::string_view request_body,
    const Servable::RunOptions& run_options, std::string* output) {
  ::google::protobuf::Arena arena;

  auto* request = ::google::protobuf::Arena::Create<ClassificationRequest>(&arena);
  TF_RETURN_IF_ERROR(FillModelSpecWithNameVersionAndLabel(
      model_name, model_version, model_version_label,
      request->mutable_model_spec()));
  TF_RETURN_IF_ERROR(FillClassificationRequestFromJson(request_body, request));

  auto* response = ::google::protobuf::Arena::Create<ClassificationResponse>(&arena);
  ServableHandle<Servable> servable;
  TF_RETURN_IF_ERROR(
      core_->GetServableHandle(request->model_spec(), &servable));
  TF_RETURN_IF_ERROR(servable->Classify(run_options, *request, response));
  TF_RETURN_IF_ERROR(
      MakeJsonFromClassificationResult(response->result(), output));
  return absl::Status();
}

absl::Status TFRTHttpRestApiHandler::ProcessRegressRequest(
    const absl::string_view model_name,
    const absl::optional<int64_t>& model_version,
    const absl::optional<absl::string_view>& model_version_label,
    const absl::string_view request_body,
    const Servable::RunOptions& run_options, std::string* output) {
  ::google::protobuf::Arena arena;

  auto* request = ::google::protobuf::Arena::Create<RegressionRequest>(&arena);
  TF_RETURN_IF_ERROR(FillModelSpecWithNameVersionAndLabel(
      model_name, model_version, model_version_label,
      request->mutable_model_spec()));
  TF_RETURN_IF_ERROR(FillRegressionRequestFromJson(request_body, request));

  auto* response = ::google::protobuf::Arena::Create<RegressionResponse>(&arena);
  ServableHandle<Servable> servable;
  TF_RETURN_IF_ERROR(
      core_->GetServableHandle(request->model_spec(), &servable));
  TF_RETURN_IF_ERROR(servable->Regress(run_options, *request, response));
  return MakeJsonFromRegressionResult(response->result(), output);
}

absl::Status TFRTHttpRestApiHandler::ProcessPredictRequest(
    const absl::string_view model_name,
    const absl::optional<int64_t>& model_version,
    const absl::optional<absl::string_view>& model_version_label,
    const absl::string_view request_body,
    const Servable::RunOptions& run_options, std::string* output) {
  ::google::protobuf::Arena arena;

  auto* request = ::google::protobuf::Arena::Create<PredictRequest>(&arena);
  TF_RETURN_IF_ERROR(FillModelSpecWithNameVersionAndLabel(
      model_name, model_version, model_version_label,
      request->mutable_model_spec()));

  JsonPredictRequestFormat format;
  TF_RETURN_IF_ERROR(FillPredictRequestFromJson(
      request_body,
      [this, request](const std::string& sig,
                      ::google::protobuf::Map<std::string, TensorInfo>* map) {
        return this->GetInfoMap(request->model_spec(), sig, map);
      },
      request, &format));

  auto* response = ::google::protobuf::Arena::Create<PredictResponse>(&arena);

  ServableHandle<Servable> servable;
  TF_RETURN_IF_ERROR(
      core_->GetServableHandle(request->model_spec(), &servable));
  TF_RETURN_IF_ERROR(servable->Predict(run_options, *request, response));

  TF_RETURN_IF_ERROR(MakeJsonFromTensors(response->outputs(), format, output));
  return absl::Status();
}

absl::Status TFRTHttpRestApiHandler::ProcessModelStatusRequest(
    const absl::string_view model_name,
    const absl::optional<int64_t>& model_version,
    const absl::optional<absl::string_view>& model_version_label,
    const Servable::RunOptions& run_options, std::string* output) {
  // We do not yet support returning status of all models
  // to be in-sync with the gRPC GetModelStatus API.
  if (model_name.empty()) {
    return errors::InvalidArgument("Missing model name in request.");
  }

  ::google::protobuf::Arena arena;

  auto* request = ::google::protobuf::Arena::Create<GetModelStatusRequest>(&arena);
  TF_RETURN_IF_ERROR(FillModelSpecWithNameVersionAndLabel(
      model_name, model_version, model_version_label,
      request->mutable_model_spec()));

  auto* response = ::google::protobuf::Arena::Create<GetModelStatusResponse>(&arena);
  TF_RETURN_IF_ERROR(
      GetModelStatusImpl::GetModelStatus(core_, *request, response));
  return ToJsonString(*response, output);
}

absl::Status TFRTHttpRestApiHandler::ProcessModelMetadataRequest(
    const absl::string_view model_name,
    const absl::optional<int64_t>& model_version,
    const absl::optional<absl::string_view>& model_version_label,
    std::string* output) {
  if (model_name.empty()) {
    return errors::InvalidArgument("Missing model name in request.");
  }

  ::google::protobuf::Arena arena;

  auto* request = ::google::protobuf::Arena::Create<GetModelMetadataRequest>(&arena);
  // We currently only support the kSignatureDef metadata field
  request->add_metadata_field(std::string(kSignatureDef));
  TF_RETURN_IF_ERROR(FillModelSpecWithNameVersionAndLabel(
      model_name, model_version, model_version_label,
      request->mutable_model_spec()));

  auto* response = ::google::protobuf::Arena::Create<GetModelMetadataResponse>(&arena);
  TF_RETURN_IF_ERROR(
      TFRTGetModelMetadataImpl::GetModelMetadata(core_, *request, response));

  return ToJsonString(*response, output);
}

absl::Status TFRTHttpRestApiHandler::GetInfoMap(
    const ModelSpec& model_spec, const std::string& signature_name,
    ::google::protobuf::Map<std::string, tensorflow::TensorInfo>* infomap) {
  ServableHandle<Servable> servable;
  TF_RETURN_IF_ERROR(core_->GetServableHandle(model_spec, &servable));
  auto& saved_model =
      down_cast<TfrtSavedModelServable*>(servable.get())->saved_model();
  const std::string& signame =
      signature_name.empty() ? kDefaultServingSignatureDefKey : signature_name;
  auto iter = saved_model.GetMetaGraphDef().signature_def().find(signame);
  if (iter == saved_model.GetMetaGraphDef().signature_def().end()) {
    return errors::InvalidArgument("Serving signature name: \"", signame,
                                   "\" not found in signature def");
  }
  *infomap = iter->second.inputs();
  return absl::Status();
}

}  // namespace serving
}  // namespace tensorflow
