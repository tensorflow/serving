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
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/threadpool_options.h"
#include "tensorflow_serving/apis/classification.pb.h"
#include "tensorflow_serving/apis/inference.pb.h"
#include "tensorflow_serving/apis/input.pb.h"
#include "tensorflow_serving/apis/model.pb.h"
#include "tensorflow_serving/apis/predict.pb.h"
#include "tensorflow_serving/apis/prediction_log.pb.h"
#include "tensorflow_serving/apis/regression.pb.h"
#include "tensorflow_serving/core/test_util/mock_session.h"
#include "tensorflow_serving/servables/tensorflow/saved_model_warmup_test_util.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_config.pb.h"

namespace tensorflow {
namespace serving {

namespace {

using test_util::MockSession;
using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;
using ::testing::SizeIs;

class SavedModelBundleWarmupOptionsTest
    : public ::testing::TestWithParam<bool> {
 public:
  bool EnableNumRequestIterations() { return GetParam(); }

  ModelWarmupOptions GetModelWarmupOptions() {
    ModelWarmupOptions options;
    if (EnableNumRequestIterations()) {
      options.mutable_num_request_iterations()->set_value(2);
    }
    return options;
  }

  int GetNumRequestIterations() {
    if (EnableNumRequestIterations()) {
      return 2;
    }
    return 1;
  }
};

TEST_P(SavedModelBundleWarmupOptionsTest, MixedWarmupData) {
  string base_path = io::JoinPath(testing::TmpDir(), "MixedWarmupData");
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
  MockSession* mock = new MockSession;
  saved_model_bundle.session.reset(mock);
  Tensor scores(DT_FLOAT, TensorShape({1, 1}));
  Tensor classes(DT_STRING, TensorShape({1, 1}));
  // Regress and Predict cases
  EXPECT_CALL(*mock, Run(_, _, SizeIs(1), _, _, _, _))
      .Times(num_warmup_records * 2 * GetNumRequestIterations())
      .WillRepeatedly(DoAll(SetArgPointee<4>(std::vector<Tensor>({scores})),
                            Return(OkStatus())));
  // Classify case
  EXPECT_CALL(*mock, Run(_, _, SizeIs(2), _, _, _, _))
      .Times(num_warmup_records * GetNumRequestIterations())
      .WillRepeatedly(
          DoAll(SetArgPointee<4>(std::vector<Tensor>({classes, scores})),
                Return(OkStatus())));
  // MultiInference case
  EXPECT_CALL(*mock, Run(_, _, SizeIs(3), _, _, _, _))
      .Times(num_warmup_records * GetNumRequestIterations())
      .WillRepeatedly(DoAll(
          SetArgPointee<4>(std::vector<Tensor>({classes, scores, scores})),
          Return(OkStatus())));
  TF_EXPECT_OK(RunSavedModelWarmup(GetModelWarmupOptions(), RunOptions(),
                                   base_path, &saved_model_bundle));
}
INSTANTIATE_TEST_SUITE_P(WarmupOptions, SavedModelBundleWarmupOptionsTest,
                         ::testing::Bool());

TEST(SavedModelBundleWarmupTest, UnsupportedLogType) {
  string base_path = io::JoinPath(testing::TmpDir(), "UnsupportedLogType");
  TF_ASSERT_OK(Env::Default()->RecursivelyCreateDir(
      io::JoinPath(base_path, kSavedModelAssetsExtraDirectory)));
  string fname = io::JoinPath(base_path, kSavedModelAssetsExtraDirectory,
                              internal::WarmupConsts::kRequestsFileName);

  std::vector<string> warmup_records;
  // Add unsupported log type
  PredictionLog prediction_log;
  PopulatePredictionLog(&prediction_log, PredictionLog::kSessionRunLog);
  warmup_records.push_back(prediction_log.SerializeAsString());
  TF_ASSERT_OK(WriteWarmupData(fname, warmup_records, 10));
  SavedModelBundle saved_model_bundle;
  AddSignatures(&saved_model_bundle.meta_graph_def);
  MockSession* mock = new MockSession;
  saved_model_bundle.session.reset(mock);
  EXPECT_CALL(*mock, Run(_, _, _, _, _, _, _))
      .WillRepeatedly(Return(OkStatus()));
  const Status status = RunSavedModelWarmup(ModelWarmupOptions(), RunOptions(),
                                            base_path, &saved_model_bundle);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(::tensorflow::error::UNIMPLEMENTED, status.code()) << status;
  EXPECT_THAT(status.ToString(),
              ::testing::HasSubstr("Unsupported log_type for warmup"));
}

}  // namespace

}  // namespace serving
}  // namespace tensorflow
