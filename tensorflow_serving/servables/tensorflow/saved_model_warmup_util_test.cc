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

#include <string>
#include <vector>

#include "google/protobuf/wrappers.pb.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/kernels/batching_util/warmup.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/threadpool_options.h"
#include "tensorflow_serving/apis/classification.pb.h"
#include "tensorflow_serving/apis/inference.pb.h"
#include "tensorflow_serving/apis/input.pb.h"
#include "tensorflow_serving/apis/model.pb.h"
#include "tensorflow_serving/apis/predict.pb.h"
#include "tensorflow_serving/apis/prediction_log.pb.h"
#include "tensorflow_serving/apis/regression.pb.h"
#include "tensorflow_serving/servables/tensorflow/saved_model_warmup_test_util.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_config.pb.h"

namespace tensorflow {
namespace serving {
namespace internal {
namespace {

constexpr absl::string_view kModelName = "/ml/owner/model";
constexpr int64_t kModelVersion = 0;
constexpr int32_t kNumWarmupThreads = 3;

class SavedModelBundleWarmupUtilTest : public ::testing::TestWithParam<bool> {
 protected:
  SavedModelBundleWarmupUtilTest() {}

  bool ParallelWarmUp() { return GetParam(); }

  ModelWarmupOptions CreateModelWarmupOptions() {
    ModelWarmupOptions options;
    if (ParallelWarmUp()) {
      options.set_model_name(std::string(kModelName));
      options.set_model_version(kModelVersion);
      options.mutable_num_model_warmup_threads()->set_value(kNumWarmupThreads);
    }
    return options;
  }

  bool LookupWarmupState() const {
    return GetGlobalWarmupStateRegistry().Lookup(
        {std::string(kModelName), kModelVersion});
  }

  void FakeRunWarmupRequest() {
    tensorflow::mutex_lock lock(mu_);
    is_model_in_warmup_state_registry_ = LookupWarmupState();
    warmup_request_counter_++;
  }

  bool is_model_in_warmup_state_registry() {
    tensorflow::mutex_lock lock(mu_);
    return is_model_in_warmup_state_registry_;
  }

  int warmup_request_counter() {
    tensorflow::mutex_lock lock(mu_);
    return warmup_request_counter_;
  }

 private:
  tensorflow::mutex mu_;
  bool is_model_in_warmup_state_registry_ = false;
  int warmup_request_counter_ = 0;
};

TEST_P(SavedModelBundleWarmupUtilTest, WarmupStateRegistration) {
  string base_path = io::JoinPath(testing::TmpDir(), "WarmupStateRegistration");
  TF_ASSERT_OK(Env::Default()->RecursivelyCreateDir(
      io::JoinPath(base_path, kSavedModelAssetsExtraDirectory)));
  string fname = io::JoinPath(base_path, kSavedModelAssetsExtraDirectory,
                              internal::WarmupConsts::kRequestsFileName);

  const int num_warmup_records = ParallelWarmUp() ? kNumWarmupThreads : 1;
  std::vector<string> warmup_records;
  TF_ASSERT_OK(
      AddMixedWarmupData(&warmup_records, {PredictionLog::kPredictLog}));
  TF_ASSERT_OK(WriteWarmupData(fname, warmup_records, num_warmup_records));

  TF_ASSERT_OK(RunSavedModelWarmup(CreateModelWarmupOptions(), base_path,
                                [this](PredictionLog prediction_log) {
                                  this->FakeRunWarmupRequest();
                                  return absl::OkStatus();
                                }));
  EXPECT_EQ(warmup_request_counter(), num_warmup_records);
  EXPECT_EQ(is_model_in_warmup_state_registry(), ParallelWarmUp());
  // The model should be unregistered from the WarmupStateRegistry after
  // warm-up.
  EXPECT_FALSE(LookupWarmupState());
}

TEST_P(SavedModelBundleWarmupUtilTest, NoWarmupDataFile) {
  string base_path = io::JoinPath(testing::TmpDir(), "NoWarmupDataFile");
  TF_ASSERT_OK(Env::Default()->RecursivelyCreateDir(
      io::JoinPath(base_path, kSavedModelAssetsExtraDirectory)));

  SavedModelBundle saved_model_bundle;
  AddSignatures(&saved_model_bundle.meta_graph_def);
  TF_EXPECT_OK(RunSavedModelWarmup(CreateModelWarmupOptions(), base_path,
                                   [this](PredictionLog prediction_log) {
                                     this->FakeRunWarmupRequest();
                                     return absl::OkStatus();
                                   }));
  EXPECT_EQ(warmup_request_counter(), 0);
}

TEST_P(SavedModelBundleWarmupUtilTest, WarmupDataFileEmpty) {
  string base_path = io::JoinPath(testing::TmpDir(), "WarmupDataFileEmpty");
  TF_ASSERT_OK(Env::Default()->RecursivelyCreateDir(
      io::JoinPath(base_path, kSavedModelAssetsExtraDirectory)));
  string fname = io::JoinPath(base_path, kSavedModelAssetsExtraDirectory,
                              internal::WarmupConsts::kRequestsFileName);

  std::vector<string> warmup_records;
  TF_ASSERT_OK(WriteWarmupData(fname, warmup_records, 0));
  SavedModelBundle saved_model_bundle;
  AddSignatures(&saved_model_bundle.meta_graph_def);
  TF_EXPECT_OK(RunSavedModelWarmup(CreateModelWarmupOptions(), base_path,
                                   [this](PredictionLog prediction_log) {
                                     this->FakeRunWarmupRequest();
                                     return absl::OkStatus();
                                   }));
  EXPECT_EQ(warmup_request_counter(), 0);
}

TEST_P(SavedModelBundleWarmupUtilTest, UnsupportedFileFormat) {
  string base_path = io::JoinPath(testing::TmpDir(), "UnsupportedFileFormat");
  TF_ASSERT_OK(Env::Default()->RecursivelyCreateDir(
      io::JoinPath(base_path, kSavedModelAssetsExtraDirectory)));
  const string fname = io::JoinPath(base_path, kSavedModelAssetsExtraDirectory,
                                    internal::WarmupConsts::kRequestsFileName);

  std::vector<string> warmup_records;
  // Add unsupported log type
  PredictionLog prediction_log;
  TF_ASSERT_OK(
      PopulatePredictionLog(&prediction_log, PredictionLog::kSessionRunLog));
  warmup_records.push_back(prediction_log.SerializeAsString());

  TF_ASSERT_OK(WriteWarmupDataAsSerializedProtos(fname, warmup_records, 10));
  SavedModelBundle saved_model_bundle;
  AddSignatures(&saved_model_bundle.meta_graph_def);
  const absl::Status status = RunSavedModelWarmup(
      CreateModelWarmupOptions(), base_path,
      [](PredictionLog prediction_log) { return absl::OkStatus(); });
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(::tensorflow::error::DATA_LOSS, status.code()) << status;
  EXPECT_THAT(status.ToString(),
              ::testing::HasSubstr(
                  "Please verify your warmup data is in TFRecord format"));
}

TEST_P(SavedModelBundleWarmupUtilTest, TooManyWarmupRecords) {
  string base_path = io::JoinPath(testing::TmpDir(), "TooManyWarmupRecords");
  TF_ASSERT_OK(Env::Default()->RecursivelyCreateDir(
      io::JoinPath(base_path, kSavedModelAssetsExtraDirectory)));
  string fname = io::JoinPath(base_path, kSavedModelAssetsExtraDirectory,
                              internal::WarmupConsts::kRequestsFileName);

  std::vector<string> warmup_records;
  TF_ASSERT_OK(AddMixedWarmupData(&warmup_records));
  TF_ASSERT_OK(WriteWarmupData(fname, warmup_records,
                               internal::WarmupConsts::kMaxNumRecords + 1));
  SavedModelBundle saved_model_bundle;
  AddSignatures(&saved_model_bundle.meta_graph_def);
  const absl::Status status = RunSavedModelWarmup(
      CreateModelWarmupOptions(), base_path,
      [](PredictionLog prediction_log) { return absl::OkStatus(); });
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(static_cast<absl::StatusCode>(absl::StatusCode::kInvalidArgument),
            status.code())
      << status;
  EXPECT_THAT(
      status.ToString(),
      ::testing::HasSubstr("Number of warmup records exceeds the maximum"));
}

TEST_P(SavedModelBundleWarmupUtilTest, UnparsableRecord) {
  string base_path = io::JoinPath(testing::TmpDir(), "UnparsableRecord");
  TF_ASSERT_OK(Env::Default()->RecursivelyCreateDir(
      io::JoinPath(base_path, kSavedModelAssetsExtraDirectory)));
  string fname = io::JoinPath(base_path, kSavedModelAssetsExtraDirectory,
                              internal::WarmupConsts::kRequestsFileName);

  std::vector<string> warmup_records = {"malformed_record"};
  TF_ASSERT_OK(WriteWarmupData(fname, warmup_records, 10));
  SavedModelBundle saved_model_bundle;
  const absl::Status status = RunSavedModelWarmup(
      CreateModelWarmupOptions(), base_path,
      [](PredictionLog prediction_log) { return absl::OkStatus(); });
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(static_cast<absl::StatusCode>(absl::StatusCode::kInvalidArgument),
            status.code())
      << status;
  EXPECT_THAT(status.ToString(),
              ::testing::HasSubstr("Failed to parse warmup record"));
}

TEST_P(SavedModelBundleWarmupUtilTest, RunFailure) {
  string base_path = io::JoinPath(testing::TmpDir(), "RunFailure");
  TF_ASSERT_OK(Env::Default()->RecursivelyCreateDir(
      io::JoinPath(base_path, kSavedModelAssetsExtraDirectory)));
  string fname = io::JoinPath(base_path, kSavedModelAssetsExtraDirectory,
                              internal::WarmupConsts::kRequestsFileName);

  int num_warmup_records = 10;
  std::vector<string> warmup_records;
  TF_ASSERT_OK(AddMixedWarmupData(&warmup_records));
  TF_ASSERT_OK(WriteWarmupData(fname, warmup_records, num_warmup_records));
  SavedModelBundle saved_model_bundle;
  AddSignatures(&saved_model_bundle.meta_graph_def);
  absl::Status status = RunSavedModelWarmup(
      CreateModelWarmupOptions(), base_path, [](PredictionLog prediction_log) {
        return errors::InvalidArgument("Run failed");
      });
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(static_cast<absl::StatusCode>(absl::StatusCode::kInvalidArgument),
            status.code())
      << status;
  EXPECT_THAT(status.ToString(), ::testing::HasSubstr("Run failed"));
}
INSTANTIATE_TEST_SUITE_P(ParallelWarmUp, SavedModelBundleWarmupUtilTest,
                         ::testing::Bool());
}  // namespace
}  // namespace internal
}  // namespace serving
}  // namespace tensorflow
