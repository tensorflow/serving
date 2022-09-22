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
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
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

class SavedModelBundleWarmupUtilTest : public ::testing::TestWithParam<bool> {
 protected:
  SavedModelBundleWarmupUtilTest() : warmup_request_counter_(0) {}
  bool ParallelWarmUp() { return GetParam(); }
  ModelWarmupOptions CreateModelWarmupOptions() {
    ModelWarmupOptions options;
    if (ParallelWarmUp()) {
      options.mutable_num_model_warmup_threads()->set_value(3);
    }
    return options;
  }
  void FakeRunWarmupRequest() { warmup_request_counter_++; }
  int warmup_request_counter_;
};

TEST_P(SavedModelBundleWarmupUtilTest, NoWarmupDataFile) {
  string base_path = io::JoinPath(testing::TmpDir(), "NoWarmupDataFile");
  TF_ASSERT_OK(Env::Default()->RecursivelyCreateDir(
      io::JoinPath(base_path, kSavedModelAssetsExtraDirectory)));

  SavedModelBundle saved_model_bundle;
  AddSignatures(&saved_model_bundle.meta_graph_def);
  TF_EXPECT_OK(RunSavedModelWarmup(CreateModelWarmupOptions(), base_path,
                                   [this](PredictionLog prediction_log) {
                                     this->FakeRunWarmupRequest();
                                     return OkStatus();
                                   }));
  EXPECT_EQ(warmup_request_counter_, 0);
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
                                     return OkStatus();
                                   }));
  EXPECT_EQ(warmup_request_counter_, 0);
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
  PopulatePredictionLog(&prediction_log, PredictionLog::kSessionRunLog);
  warmup_records.push_back(prediction_log.SerializeAsString());

  TF_ASSERT_OK(WriteWarmupDataAsSerializedProtos(fname, warmup_records, 10));
  SavedModelBundle saved_model_bundle;
  AddSignatures(&saved_model_bundle.meta_graph_def);
  const Status status = RunSavedModelWarmup(
      CreateModelWarmupOptions(), base_path,
      [](PredictionLog prediction_log) { return OkStatus(); });
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
  AddMixedWarmupData(&warmup_records);
  TF_ASSERT_OK(WriteWarmupData(fname, warmup_records,
                               internal::WarmupConsts::kMaxNumRecords + 1));
  SavedModelBundle saved_model_bundle;
  AddSignatures(&saved_model_bundle.meta_graph_def);
  const Status status = RunSavedModelWarmup(
      CreateModelWarmupOptions(), base_path,
      [](PredictionLog prediction_log) { return OkStatus(); });
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, status.code()) << status;
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
  const Status status = RunSavedModelWarmup(
      CreateModelWarmupOptions(), base_path,
      [](PredictionLog prediction_log) { return OkStatus(); });
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, status.code()) << status;
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
  AddMixedWarmupData(&warmup_records);
  TF_ASSERT_OK(WriteWarmupData(fname, warmup_records, num_warmup_records));
  SavedModelBundle saved_model_bundle;
  AddSignatures(&saved_model_bundle.meta_graph_def);
  Status status = RunSavedModelWarmup(
      CreateModelWarmupOptions(), base_path, [](PredictionLog prediction_log) {
        return errors::InvalidArgument("Run failed");
      });
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, status.code()) << status;
  EXPECT_THAT(status.ToString(), ::testing::HasSubstr("Run failed"));
}
INSTANTIATE_TEST_SUITE_P(ParallelWarmUp, SavedModelBundleWarmupUtilTest,
                         ::testing::Bool());
}  // namespace
}  // namespace internal
}  // namespace serving
}  // namespace tensorflow
