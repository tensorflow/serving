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

#ifndef THIRD_PARTY_TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_SAVED_MODEL_WARMUP_TEST_UTIL_H_
#define THIRD_PARTY_TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_SAVED_MODEL_WARMUP_TEST_UTIL_H_

#include <string>

#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow_serving/apis/classification.pb.h"
#include "tensorflow_serving/apis/inference.pb.h"
#include "tensorflow_serving/apis/input.pb.h"
#include "tensorflow_serving/apis/model.pb.h"
#include "tensorflow_serving/apis/predict.pb.h"
#include "tensorflow_serving/apis/prediction_log.pb.h"
#include "tensorflow_serving/apis/regression.pb.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_config.pb.h"

namespace tensorflow {
namespace serving {

void PopulateInferenceTask(const string& model_name,
                           const string& signature_name,
                           const string& method_name, InferenceTask* task);

void PopulateMultiInferenceRequest(MultiInferenceRequest* request);

void PopulatePredictRequest(PredictRequest* request);

void PopulateClassificationRequest(ClassificationRequest* request);

void PopulateRegressionRequest(RegressionRequest* request);

void PopulatePredictionLog(PredictionLog* prediction_log,
                           PredictionLog::LogTypeCase log_type);

Status WriteWarmupData(const string& fname,
                       const std::vector<string>& warmup_records,
                       int num_warmup_records);

Status WriteWarmupDataAsSerializedProtos(
    const string& fname, const std::vector<string>& warmup_records,
    int num_warmup_records);

void AddMixedWarmupData(
    std::vector<string>* warmup_records,
    const std::vector<PredictionLog::LogTypeCase>& log_types = {
        PredictionLog::kRegressLog, PredictionLog::kClassifyLog,
        PredictionLog::kPredictLog, PredictionLog::kMultiInferenceLog});

// Creates a test SignatureDef with the given parameters
SignatureDef CreateSignatureDef(const string& method_name,
                                const std::vector<string>& input_names,
                                const std::vector<string>& output_names);

void AddSignatures(MetaGraphDef* meta_graph_def);

}  // namespace serving
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_SAVED_MODEL_WARMUP_TEST_UTIL_H_
