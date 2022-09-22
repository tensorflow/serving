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
#include "google/protobuf/arena.h"
#include "google/protobuf/util/json_util.h"
#include "absl/strings/escaping.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/threadpool_options.h"
#include "tensorflow_serving/apis/model.pb.h"
#include "tensorflow_serving/apis/predict.pb.h"
#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/model_servers/get_model_status_impl.h"
#include "tensorflow_serving/model_servers/http_rest_api_util.h"
#include "tensorflow_serving/model_servers/server_core.h"
#include "tensorflow_serving/servables/tensorflow/classification_service.h"
#include "tensorflow_serving/servables/tensorflow/get_model_metadata_impl.h"
#include "tensorflow_serving/servables/tensorflow/predict_impl.h"
#include "tensorflow_serving/servables/tensorflow/regression_service.h"
#include "tensorflow_serving/util/json_tensor.h"

namespace tensorflow {
namespace serving {

using tensorflow::serving::ServerCore;
using tensorflow::serving::TensorflowPredictor;

const char* const HttpRestApiHandler::kPathRegex = kHTTPRestApiHandlerPathRegex;

HttpRestApiHandler::HttpRestApiHandler(int timeout_in_ms, ServerCore* core)
    : run_options_(), core_(core), predictor_(new TensorflowPredictor()) {
  if (timeout_in_ms > 0) {
    run_options_.set_timeout_in_ms(timeout_in_ms);
  }
}

HttpRestApiHandler::~HttpRestApiHandler() {}

Status HttpRestApiHandler::ProcessRequest(
    const absl::string_view http_method, const absl::string_view request_path,
    const absl::string_view request_body,
    std::vector<std::pair<string, string>>* headers, string* model_name,
    string* method, string* output) {
  headers->clear();
  output->clear();
  AddHeaders(headers);
  string model_subresource;
  Status status = errors::InvalidArgument("Malformed request: ", http_method,
                                          " ", request_path);
  absl::optional<int64_t> model_version;
  absl::optional<string> model_version_label;
  bool parse_successful;

  TF_RETURN_IF_ERROR(ParseModelInfo(
      http_method, request_path, model_name, &model_version,
      &model_version_label, method, &model_subresource, &parse_successful));

  // Dispatch request to appropriate processor
  if (http_method == "POST" && parse_successful) {
    if (*method == "classify") {
      status =
          ProcessClassifyRequest(*model_name, model_version,
                                 model_version_label, request_body, output);
    } else if (*method == "regress") {
      status = ProcessRegressRequest(*model_name, model_version,
                                     model_version_label, request_body, output);
    } else if (*method == "predict") {
      status = ProcessPredictRequest(*model_name, model_version,
                                     model_version_label, request_body, output);
    }
  } else if (http_method == "GET" && parse_successful) {
    if (!model_subresource.empty() && model_subresource == "metadata") {
      status = ProcessModelMetadataRequest(*model_name, model_version,
                                           model_version_label, output);
    } else {
      status = ProcessModelStatusRequest(*model_name, model_version,
                                         model_version_label, output);
    }
  }

  MakeJsonFromStatus(status, output);
  return status;
}

Status HttpRestApiHandler::ProcessClassifyRequest(
    const absl::string_view model_name,
    const absl::optional<int64_t>& model_version,
    const absl::optional<absl::string_view>& model_version_label,
    const absl::string_view request_body, string* output) {
  ::google::protobuf::Arena arena;

  auto* request = ::google::protobuf::Arena::CreateMessage<ClassificationRequest>(&arena);
  TF_RETURN_IF_ERROR(FillModelSpecWithNameVersionAndLabel(
      model_name, model_version, model_version_label,
      request->mutable_model_spec()));
  TF_RETURN_IF_ERROR(FillClassificationRequestFromJson(request_body, request));

  auto* response =
      ::google::protobuf::Arena::CreateMessage<ClassificationResponse>(&arena);
  TF_RETURN_IF_ERROR(TensorflowClassificationServiceImpl::Classify(
      run_options_, core_, thread::ThreadPoolOptions(), *request, response));
  TF_RETURN_IF_ERROR(
      MakeJsonFromClassificationResult(response->result(), output));
  return OkStatus();
}

Status HttpRestApiHandler::ProcessRegressRequest(
    const absl::string_view model_name,
    const absl::optional<int64_t>& model_version,
    const absl::optional<absl::string_view>& model_version_label,
    const absl::string_view request_body, string* output) {
  ::google::protobuf::Arena arena;

  auto* request = ::google::protobuf::Arena::CreateMessage<RegressionRequest>(&arena);
  TF_RETURN_IF_ERROR(FillModelSpecWithNameVersionAndLabel(
      model_name, model_version, model_version_label,
      request->mutable_model_spec()));
  TF_RETURN_IF_ERROR(FillRegressionRequestFromJson(request_body, request));

  auto* response = ::google::protobuf::Arena::CreateMessage<RegressionResponse>(&arena);
  TF_RETURN_IF_ERROR(TensorflowRegressionServiceImpl::Regress(
      run_options_, core_, thread::ThreadPoolOptions(), *request, response));
  TF_RETURN_IF_ERROR(MakeJsonFromRegressionResult(response->result(), output));
  return OkStatus();
}

Status HttpRestApiHandler::ProcessPredictRequest(
    const absl::string_view model_name,
    const absl::optional<int64_t>& model_version,
    const absl::optional<absl::string_view>& model_version_label,
    const absl::string_view request_body, string* output) {
  ::google::protobuf::Arena arena;

  auto* request = ::google::protobuf::Arena::CreateMessage<PredictRequest>(&arena);
  TF_RETURN_IF_ERROR(FillModelSpecWithNameVersionAndLabel(
      model_name, model_version, model_version_label,
      request->mutable_model_spec()));

  JsonPredictRequestFormat format;
  TF_RETURN_IF_ERROR(FillPredictRequestFromJson(
      request_body,
      [this, request](const string& sig,
                      ::google::protobuf::Map<string, TensorInfo>* map) {
        return this->GetInfoMap(request->model_spec(), sig, map);
      },
      request, &format));

  auto* response = ::google::protobuf::Arena::CreateMessage<PredictResponse>(&arena);
  TF_RETURN_IF_ERROR(
      predictor_->Predict(run_options_, core_, *request, response));
  TF_RETURN_IF_ERROR(MakeJsonFromTensors(response->outputs(), format, output));
  return OkStatus();
}

Status HttpRestApiHandler::ProcessModelStatusRequest(
    const absl::string_view model_name,
    const absl::optional<int64_t>& model_version,
    const absl::optional<absl::string_view>& model_version_label,
    string* output) {
  // We do not yet support returning status of all models
  // to be in-sync with the gRPC GetModelStatus API.
  if (model_name.empty()) {
    return errors::InvalidArgument("Missing model name in request.");
  }

  ::google::protobuf::Arena arena;

  auto* request = ::google::protobuf::Arena::CreateMessage<GetModelStatusRequest>(&arena);
  TF_RETURN_IF_ERROR(FillModelSpecWithNameVersionAndLabel(
      model_name, model_version, model_version_label,
      request->mutable_model_spec()));

  auto* response =
      ::google::protobuf::Arena::CreateMessage<GetModelStatusResponse>(&arena);
  TF_RETURN_IF_ERROR(
      GetModelStatusImpl::GetModelStatus(core_, *request, response));
  return ToJsonString(*response, output);
}

Status HttpRestApiHandler::ProcessModelMetadataRequest(
    const absl::string_view model_name,
    const absl::optional<int64_t>& model_version,
    const absl::optional<absl::string_view>& model_version_label,
    string* output) {
  if (model_name.empty()) {
    return errors::InvalidArgument("Missing model name in request.");
  }

  ::google::protobuf::Arena arena;

  auto* request =
      ::google::protobuf::Arena::CreateMessage<GetModelMetadataRequest>(&arena);
  // We currently only support the kSignatureDef metadata field
  request->add_metadata_field(GetModelMetadataImpl::kSignatureDef);
  TF_RETURN_IF_ERROR(FillModelSpecWithNameVersionAndLabel(
      model_name, model_version, model_version_label,
      request->mutable_model_spec()));

  auto* response =
      ::google::protobuf::Arena::CreateMessage<GetModelMetadataResponse>(&arena);
  TF_RETURN_IF_ERROR(
      GetModelMetadataImpl::GetModelMetadata(core_, *request, response));
  return ToJsonString(*response, output);
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
  return OkStatus();
}

}  // namespace serving
}  // namespace tensorflow
