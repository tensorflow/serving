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

#include "tensorflow_serving/servables/tensorflow/saved_model_warmup.h"

#include "google/protobuf/wrappers.pb.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/monitoring/sampler.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow_serving/apis/prediction_log.pb.h"
#include "tensorflow_serving/servables/tensorflow/classifier.h"
#include "tensorflow_serving/servables/tensorflow/multi_inference.h"
#include "tensorflow_serving/servables/tensorflow/predict_util.h"
#include "tensorflow_serving/servables/tensorflow/regressor.h"
#include "tensorflow_serving/servables/tensorflow/util.h"

namespace tensorflow {
namespace serving {

namespace {

Status RunWarmupRequest(const PredictionLog& warmup_record,
                        const RunOptions& run_options,
                        const MetaGraphDef& meta_graph_def, Session* session) {
  switch (warmup_record.log_type_case()) {
    case PredictionLog::kRegressLog: {
      RegressionResponse response;
      TF_RETURN_IF_ERROR(RunRegress(run_options, meta_graph_def, {}, session,
                                    warmup_record.regress_log().request(),
                                    &response));
    } break;
    case PredictionLog::kClassifyLog: {
      ClassificationResponse response;
      TF_RETURN_IF_ERROR(RunClassify(run_options, meta_graph_def, {}, session,
                                     warmup_record.classify_log().request(),
                                     &response));
    } break;
    case PredictionLog::kPredictLog: {
      PredictResponse response;
      TF_RETURN_IF_ERROR(RunPredict(run_options, meta_graph_def, {}, session,
                                    warmup_record.predict_log().request(),
                                    &response));
    } break;
    case PredictionLog::kMultiInferenceLog: {
      MultiInferenceResponse response;
      TF_RETURN_IF_ERROR(RunMultiInference(
          run_options, meta_graph_def, {}, session,
          warmup_record.multi_inference_log().request(), &response));
    } break;
    case PredictionLog::kSessionRunLog:
      return errors::Unimplemented(strings::StrCat(
          "Unsupported log_type for warmup: ", warmup_record.log_type_case()));
    default:
      break;
  }
  return OkStatus();
}

}  // namespace

Status RunSavedModelWarmup(const ModelWarmupOptions& model_warmup_options,
                           const RunOptions& run_options,
                           const string& export_dir, SavedModelBundle* bundle) {
  return internal::RunSavedModelWarmup(
      model_warmup_options, export_dir, [&](PredictionLog prediction_log) {
        return RunWarmupRequest(prediction_log, run_options,
                                bundle->meta_graph_def, bundle->GetSession());
      });
}

}  // namespace serving
}  // namespace tensorflow
