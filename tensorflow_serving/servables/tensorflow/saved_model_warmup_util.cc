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

#include "tensorflow_serving/servables/tensorflow/saved_model_warmup_util.h"

#include "google/protobuf/wrappers.pb.h"
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/monitoring/sampler.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow_serving/util/threadpool_executor.h"

namespace tensorflow {
namespace serving {
namespace internal {
namespace {

auto* model_warm_up_latency = monitoring::Sampler<2>::New(
    {
        "/tensorflow/serving/model_warmup_latency",
        "Distribution of wall time (in microseconds) for warming up the model.",
        "model_path",
        "status",
    },  // Scale of 10, power of 1.8 with bucket count 33 (~20 minutes).
    monitoring::Buckets::Exponential(10, 1.8, 33));

uint64_t GetLatencyMicroseconds(const uint64_t start_microseconds) {
  const uint64_t end_microseconds = EnvTime::NowMicros();
  // Avoid clock skew.
  if (end_microseconds < start_microseconds) return 0;
  return end_microseconds - start_microseconds;
}

}  // namespace

constexpr char WarmupConsts::kRequestsFileName[];
constexpr int WarmupConsts::kMaxNumRecords;

Status RunSavedModelWarmup(
    const ModelWarmupOptions& model_warmup_options, const string export_dir,
    std::function<Status(PredictionLog)> warmup_request_executor) {
  const uint64_t start_microseconds = EnvTime::NowMicros();
  const string warmup_path =
      io::JoinPath(export_dir, kSavedModelAssetsExtraDirectory,
                   WarmupConsts::kRequestsFileName);
  if (!tensorflow::Env::Default()->FilesExist({warmup_path}, nullptr)) {
    LOG(INFO) << "No warmup data file found at " << warmup_path;
    // Having warmup data is optional, return OK
    return OkStatus();
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

  int num_model_warmup_threads =
      model_warmup_options.has_num_model_warmup_threads()
          ? std::max(model_warmup_options.num_model_warmup_threads().value(), 1)
          : 1;
  std::unique_ptr<tensorflow::io::SequentialRecordReader> tf_record_file_reader;
  Status status;
  int num_warmup_records = 0;
  if (num_model_warmup_threads <= 1) {
    tf_record_file_reader.reset(
        new tensorflow::io::SequentialRecordReader(tf_record_file.get()));
    tstring record;
    status = tf_record_file_reader->ReadRecord(&record);
    tensorflow::serving::PredictionLog prediction_log;
    while (status.ok()) {
      if (!prediction_log.ParseFromArray(record.data(), record.size())) {
        return errors::InvalidArgument(strings::StrCat(
            "Failed to parse warmup record: ", record, " from ", warmup_path));
      }

      for (int i = 0; i < num_request_iterations; ++i) {
        TF_RETURN_IF_ERROR(warmup_request_executor(prediction_log));
      }
      ++num_warmup_records;
      if (num_warmup_records > WarmupConsts::kMaxNumRecords) {
        return errors::InvalidArgument(
            "Number of warmup records exceeds the maximum (",
            WarmupConsts::kMaxNumRecords, ") at ", warmup_path);
      }
      status = tf_record_file_reader->ReadRecord(&record);
    }
  } else {
    struct SharedState {
      ::tensorflow::mutex mu;
      int num_thread_task_done ABSL_GUARDED_BY(mu){0};
      int num_warmup_records ABSL_GUARDED_BY(mu){0};
      ::tensorflow::Status warm_up_status ABSL_GUARDED_BY(mu);
      // Condition variable to wait until all scheduled warmup tasks are
      // executed.
      ::tensorflow::condition_variable done ABSL_GUARDED_BY(mu);
      std::unique_ptr<tensorflow::io::SequentialRecordReader>
          tf_record_file_reader ABSL_GUARDED_BY(mu);
    };
    const auto state = std::make_shared<SharedState>();

    std::unique_ptr<Executor> executor;
    executor.reset(new ThreadPoolExecutor(Env::Default(), "Warmup_ThreadPool",
                                          num_model_warmup_threads));
    {
      ::tensorflow::mutex_lock lock(state->mu);
      state->tf_record_file_reader.reset(
          new tensorflow::io::SequentialRecordReader(tf_record_file.get()));
    }
    for (int i = 0; i < num_model_warmup_threads; ++i) {
      executor->Schedule([state, num_request_iterations,
                          warmup_request_executor, warmup_path,
                          num_model_warmup_threads]() {
        Status status = OkStatus();
        while (status.ok()) {
          tstring record;
          Status execution_status;
          tensorflow::serving::PredictionLog prediction_log;
          {
            ::tensorflow::mutex_lock lock(state->mu);
            if (state->num_warmup_records > WarmupConsts::kMaxNumRecords) {
              state->warm_up_status = errors::InvalidArgument(
                  "Number of warmup records exceeds the maximum (",
                  WarmupConsts::kMaxNumRecords, ") at ", warmup_path);
              break;
            }
            if (!state->warm_up_status.ok()) {
              break;
            }
            execution_status =
                state->tf_record_file_reader->ReadRecord(&record);
            if (!execution_status.ok()) {
              state->warm_up_status = execution_status;
              break;
            }
            if (!prediction_log.ParseFromArray(record.data(), record.size())) {
              state->warm_up_status = errors::InvalidArgument(
                  strings::StrCat("Failed to parse warmup record: ", record,
                                  " from ", warmup_path));
              break;
            }
          }
          for (int i = 0; i < num_request_iterations; ++i) {
            execution_status = warmup_request_executor(prediction_log);
            if (!execution_status.ok()) {
              ::tensorflow::mutex_lock lock(state->mu);
              state->warm_up_status = execution_status;
              break;
            }
          }
          ::tensorflow::mutex_lock lock(state->mu);
          ++state->num_warmup_records;
          status = state->warm_up_status;
        }
        ::tensorflow::mutex_lock lock(state->mu);
        if (++state->num_thread_task_done == num_model_warmup_threads) {
          state->done.notify_one();
        }
      });
    }
    // Wait until all scheduled work are done.
    ::tensorflow::mutex_lock lock(state->mu);
    while (state->num_thread_task_done < num_model_warmup_threads) {
      state->done.wait(lock);
    }
    status = state->warm_up_status;
    num_warmup_records = state->num_warmup_records;
  }
  // OUT_OF_RANGE error means EOF was reached, re-write it to OK; in this way
  // the 'model_warm_up_latency' metric below records OK upon successful
  // warm-up.
  if (errors::IsOutOfRange(status)) {
    status = OkStatus();
  }

  const auto warmup_latency = GetLatencyMicroseconds(start_microseconds);
  model_warm_up_latency->GetCell(export_dir, status.ToString())
      ->Add(warmup_latency);

  if (errors::IsDataLoss(status)) {
    return errors::DataLoss(
        status.error_message(),
        ". Please verify your warmup data is in TFRecord format.");
  }

  TF_RETURN_IF_ERROR(status);

  LOG(INFO) << "Finished reading warmup data for model at " << warmup_path
            << ". Number of warmup records read: " << num_warmup_records
            << ". Elapsed time (microseconds): " << warmup_latency << ".";
  return OkStatus();
}

}  // namespace internal
}  // namespace serving
}  // namespace tensorflow
