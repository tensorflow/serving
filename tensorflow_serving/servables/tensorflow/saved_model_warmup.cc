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

auto* model_warm_up_latency = monitoring::Sampler<2>::New(
    {
        "/tensorflow/serving/model_warmup_latency",
        "Distribution of wall time (in microseconds) for warming up the model.",
        "model_path",
        "status",
    },  // Scale of 10, power of 1.8 with bucket count 33 (~20 minutes).
    monitoring::Buckets::Exponential(10, 1.8, 33));

uint64 GetLatencyMicroseconds(const uint64 start_microseconds) {
  const uint64 end_microseconds = Env::Default()->NowMicros();
  // Avoid clock skew.
  if (end_microseconds < start_microseconds) return 0;
  return end_microseconds - start_microseconds;
}

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
  return Status::OK();
}

}  // namespace

constexpr char WarmupConsts::kRequestsFileName[];
constexpr int WarmupConsts::kMaxNumRecords;

Status RunSavedModelWarmup(const ModelWarmupOptions& model_warmup_options,
                           const RunOptions& run_options,
                           const string& export_dir, SavedModelBundle* bundle) {
  const uint64 start_microseconds = Env::Default()->NowMicros();
  const string warmup_path =
      io::JoinPath(export_dir, kSavedModelAssetsExtraDirectory,
                   WarmupConsts::kRequestsFileName);
  if (!tensorflow::Env::Default()->FilesExist({warmup_path}, nullptr)) {
    LOG(INFO) << "No warmup data file found at " << warmup_path;
    // Having warmup data is optional, return OK
    return Status::OK();
  }

  const int num_request_iterations = [&]() {
    if (model_warmup_options.has_num_request_iterations()) {
      return model_warmup_options.num_request_iterations().value();
    }
    // Default of 1.
    return 1;
  }();
  LOG(INFO) << "Starting to read warmup data for model at " << warmup_path
            << " with model-warmup-options "
            << model_warmup_options.DebugString();
  std::unique_ptr<tensorflow::RandomAccessFile> tf_record_file;
  TF_RETURN_IF_ERROR(tensorflow::Env::Default()->NewRandomAccessFile(
      warmup_path, &tf_record_file));

  std::unique_ptr<tensorflow::io::SequentialRecordReader> tf_record_file_reader;
  tf_record_file_reader.reset(
      new tensorflow::io::SequentialRecordReader(tf_record_file.get()));
  int num_warmup_records = 0;
  string record;
  Status status = tf_record_file_reader->ReadRecord(&record);
  tensorflow::serving::PredictionLog prediction_log;
  while (status.ok()) {
    if (!prediction_log.ParseFromString(record)) {
      return errors::InvalidArgument(strings::StrCat(
          "Failed to parse warmup record: ", record, " from ", warmup_path));
    }

    for (int i = 0; i < num_request_iterations; ++i) {
      TF_RETURN_IF_ERROR(RunWarmupRequest(prediction_log, run_options,
                                          bundle->meta_graph_def,
                                          bundle->session.get()));
    }
    ++num_warmup_records;
    if (num_warmup_records > WarmupConsts::kMaxNumRecords) {
      return errors::InvalidArgument(
          "Number of warmup records exceeeds the maximum (",
          WarmupConsts::kMaxNumRecords, ") at ", warmup_path);
    }
    status = tf_record_file_reader->ReadRecord(&record);
  }

  const auto warmup_latency = GetLatencyMicroseconds(start_microseconds);
  model_warm_up_latency->GetCell(export_dir, status.ToString())
      ->Add(warmup_latency);

  if (errors::IsDataLoss(status)) {
    return errors::DataLoss(
        status.error_message(),
        ". Please verify your warmup data is in TFRecord format.");
  }

  // OUT_OF_RANGE error means EOF was reached, do not return error in this case
  if (!errors::IsOutOfRange(status)) {
    return status;
  }

  LOG(INFO) << "Finished reading warmup data for model at " << warmup_path
            << ". Number of warmup records read: " << num_warmup_records
            << ". Elapsed time (microseconds): " << warmup_latency << ".";
  return Status::OK();
}

}  // namespace serving
}  // namespace tensorflow
