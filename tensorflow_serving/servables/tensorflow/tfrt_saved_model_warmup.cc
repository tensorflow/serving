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

#include "tensorflow_serving/servables/tensorflow/tfrt_saved_model_warmup.h"

#include <string>

#include "google/protobuf/wrappers.pb.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/monitoring/sampler.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow_serving/apis/classification.pb.h"
#include "tensorflow_serving/apis/inference.pb.h"
#include "tensorflow_serving/apis/predict.pb.h"
#include "tensorflow_serving/apis/prediction_log.pb.h"
#include "tensorflow_serving/apis/regression.pb.h"
#include "tensorflow_serving/servables/tensorflow/tfrt_classifier.h"
#include "tensorflow_serving/servables/tensorflow/tfrt_multi_inference.h"
#include "tensorflow_serving/servables/tensorflow/tfrt_predict_util.h"
#include "tensorflow_serving/servables/tensorflow/tfrt_regressor.h"
#include "tensorflow_serving/servables/tensorflow/util.h"

namespace tensorflow {
namespace serving {
namespace {

absl::Status RunWarmupRequest(const PredictionLog& warmup_record,
                              const tfrt::SavedModel::RunOptions& run_options,
                              int lazy_init_threshold,
                              bool skip_warmup_requests_if_initialized,
                              tfrt::SavedModel* saved_model) {
  // If the signature defs are already initilized and
  // skip_warmup_requests_if_initialized is set to true, skip executing warmup
  // requests. We always execute MultiInference warmup requests as it will
  // trigger the compilation and initialization for combination of signature
  // defs, which won't be triggered during model loading.
  if (skip_warmup_requests_if_initialized &&
      saved_model->GetMetaGraphDef().signature_def_size() <=
          lazy_init_threshold &&
      warmup_record.log_type_case() != PredictionLog::kMultiInferenceLog) {
    return absl::OkStatus();
  }

  switch (warmup_record.log_type_case()) {
    case PredictionLog::kPredictLog: {
      PredictResponse response;
      TF_RETURN_IF_ERROR(RunPredict(run_options, {}, saved_model,
                                    warmup_record.predict_log().request(),
                                    &response));
    } break;
    case PredictionLog::kPredictStreamedLog: {
      if (warmup_record.predict_streamed_log().request_size() == 0) {
        return absl::InvalidArgumentError(absl::StrCat(
            "predict_streamed_log does not contain any requests."));
      }
      if (warmup_record.predict_streamed_log().request_size() > 1) {
        return absl::InvalidArgumentError(
            absl::StrCat("predict_streamed_log contains more than one request, "
                         "which is not supported by PredictStreamed."));
      }
      PredictResponse response;
      auto run_opts = run_options;
      run_opts.streamed_output_callback =
          [](absl::flat_hash_map<std::string, tensorflow::Tensor>) {};
      TF_RETURN_IF_ERROR(RunPredict(
          run_opts, {}, saved_model,
          warmup_record.predict_streamed_log().request(0), &response));
    } break;
    case PredictionLog::kClassifyLog: {
      ClassificationResponse response;
      TF_RETURN_IF_ERROR(RunClassify(run_options, {}, saved_model,
                                     warmup_record.classify_log().request(),
                                     &response));
      break;
    }
    case PredictionLog::kRegressLog: {
      RegressionResponse response;
      TF_RETURN_IF_ERROR(RunRegress(run_options, {}, saved_model,
                                    warmup_record.regress_log().request(),
                                    &response));
      break;
    }
    case PredictionLog::kMultiInferenceLog: {
      MultiInferenceResponse response;
      TF_RETURN_IF_ERROR(RunMultiInference(
          run_options, {}, saved_model,
          warmup_record.multi_inference_log().request(), &response));
      break;
    }
    default:
      return errors::Unimplemented(strings::StrCat(
          "Unsupported log_type for warmup: ", warmup_record.log_type_case()));
      break;
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status RunSavedModelWarmup(const ModelWarmupOptions& model_warmup_options,
                                 const string& export_dir,
                                 int lazy_init_threshold,
                                 bool skip_warmup_requests_if_initialized,
                                 tfrt::SavedModel* saved_model) {
  tfrt::SavedModel::RunOptions run_options;  // Default RunOptions.
  return internal::RunSavedModelWarmup(
      model_warmup_options, export_dir, [&](PredictionLog prediction_log) {
        return RunWarmupRequest(
            prediction_log, run_options, lazy_init_threshold,
            skip_warmup_requests_if_initialized, saved_model);
      });
}

}  // namespace serving
}  // namespace tensorflow
