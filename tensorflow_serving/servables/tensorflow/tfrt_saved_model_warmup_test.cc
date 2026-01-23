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
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/tfrt/graph_executor/graph_execution_options.h"
#include "tensorflow/core/tfrt/saved_model/saved_model.h"
#include "tensorflow/core/tfrt/utils/tensor_util.h"
#include "tsl/platform/path.h"
#include "tensorflow_serving/apis/classification.pb.h"
#include "tensorflow_serving/apis/inference.pb.h"
#include "tensorflow_serving/apis/input.pb.h"
#include "tensorflow_serving/apis/model.pb.h"
#include "tensorflow_serving/apis/predict.pb.h"
#include "tensorflow_serving/apis/prediction_log.pb.h"
#include "tensorflow_serving/apis/regression.pb.h"
#include "tensorflow_serving/servables/tensorflow/saved_model_warmup_test_util.h"
#include "tensorflow_serving/servables/tensorflow/test_util/mock_tfrt_saved_model.h"

namespace tensorflow {
namespace serving {
namespace {

using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::ReturnRef;
using ::testing::WithArgs;

class TFRTSavedModelWarmupOptionsTest : public ::testing::TestWithParam<bool> {
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

TEST_P(TFRTSavedModelWarmupOptionsTest, MixedWarmupData) {
  string base_path = io::JoinPath(testing::TmpDir(), "MixedWarmupData");
  TF_ASSERT_OK(Env::Default()->RecursivelyCreateDir(
      io::JoinPath(base_path, kSavedModelAssetsExtraDirectory)));
  string fname = io::JoinPath(base_path, kSavedModelAssetsExtraDirectory,
                              internal::WarmupConsts::kRequestsFileName);

  int num_warmup_records = 10;
  std::vector<string> warmup_records;
  TF_ASSERT_OK(AddMixedWarmupData(&warmup_records));
  TF_ASSERT_OK(WriteWarmupData(fname, warmup_records, num_warmup_records));

  std::unique_ptr<test_util::MockSavedModel> saved_model(
      (new test_util::MockSavedModel()));
  tfrt::internal::Signature predict_signature;
  predict_signature.input_names = {kPredictInputs};
  tfrt::TensorSpec spec(tensorflow::DT_STRING);
  predict_signature.input_specs = {spec};
  predict_signature.output_names = {kPredictOutputs};
  tfrt::FunctionMetadata predict_function_metadata(&predict_signature);
  EXPECT_CALL(*saved_model, GetFunctionMetadata(kPredictMethodName))
      .WillRepeatedly(Return(predict_function_metadata));

  tfrt::internal::Signature classify_signature;
  classify_signature.input_names = {kClassifyInputs};
  classify_signature.output_names = {kClassifyOutputClasses,
                                     kClassifyOutputScores};
  tfrt::FunctionMetadata classify_function_metadata(&classify_signature);
  EXPECT_CALL(*saved_model, GetFunctionMetadata(kClassifyMethodName))
      .WillRepeatedly(Return(classify_function_metadata));

  tfrt::internal::Signature regress_signature;
  regress_signature.input_names = {kRegressInputs};
  regress_signature.output_names = {kRegressOutputs};
  tfrt::FunctionMetadata regress_function_metadata(&regress_signature);
  EXPECT_CALL(*saved_model, GetFunctionMetadata(kRegressMethodName))
      .WillRepeatedly(Return(regress_function_metadata));

  MetaGraphDef meta_graph_def;
  AddSignatures(&meta_graph_def);
  EXPECT_CALL(*saved_model, GetMetaGraphDef())
      .WillRepeatedly(ReturnRef(meta_graph_def));

  Tensor scores(DT_FLOAT, TensorShape({1, 1}));
  Tensor classes(DT_STRING, TensorShape({1, 1}));

  EXPECT_CALL(*saved_model, Run(_, ::testing::Eq(kPredictMethodName),
                                ::testing::An<absl::Span<const Tensor>>(), _))
      .Times(num_warmup_records * GetNumRequestIterations())
      .WillRepeatedly(
          DoAll(WithArgs<3>([&](std::vector<Tensor>* output_tensors) {
                  output_tensors->push_back(scores);
                }),
                Return(absl::OkStatus())));

  EXPECT_CALL(*saved_model, Run(_, ::testing::Eq(kRegressMethodName),
                                ::testing::An<absl::Span<const Tensor>>(), _))
      .Times(num_warmup_records * GetNumRequestIterations())
      .WillRepeatedly(
          DoAll(WithArgs<3>([&](std::vector<Tensor>* output_tensors) {
                  output_tensors->push_back(scores);
                }),
                Return(absl::OkStatus())));

  EXPECT_CALL(*saved_model, Run(_, ::testing::Eq(kClassifyMethodName),
                                ::testing::An<absl::Span<const Tensor>>(), _))
      .Times(num_warmup_records * GetNumRequestIterations())
      .WillRepeatedly(
          DoAll(WithArgs<3>([&](std::vector<Tensor>* output_tensors) {
                  output_tensors->push_back(classes);
                  output_tensors->push_back(scores);
                }),
                Return(absl::OkStatus())));

  EXPECT_CALL(*saved_model, RunMultipleSignatures(_, _, _, _))
      .Times(num_warmup_records * GetNumRequestIterations())
      .WillRepeatedly(DoAll(
          WithArgs<3>([&](std::vector<std::vector<Tensor>>* output_tensors) {
            output_tensors->resize(2);
            (*output_tensors)[0].push_back(scores);
            (*output_tensors)[1].push_back(classes);
            (*output_tensors)[1].push_back(scores);
          }),
          Return(absl::OkStatus())));

  TF_EXPECT_OK(RunSavedModelWarmup(GetModelWarmupOptions(), base_path,
                                   /*lazy_init_threshold=*/0,
                                   /*skip_warmup_requests_if_initialized=*/true,
                                   saved_model.get()));
}

TEST_P(TFRTSavedModelWarmupOptionsTest, PredictStreamedWarmupData) {
  std::string base_path =
      tsl::io::JoinPath(testing::TmpDir(), "PredictStreamedWarmupData");
  TF_ASSERT_OK(Env::Default()->RecursivelyCreateDir(
      tsl::io::JoinPath(base_path, kSavedModelAssetsExtraDirectory)));
  std::string fname =
      tsl::io::JoinPath(base_path, kSavedModelAssetsExtraDirectory,
                        internal::WarmupConsts::kRequestsFileName);

  int num_warmup_records = 10;
  std::vector<std::string> warmup_records;
  TF_ASSERT_OK(
      AddToWarmupData(&warmup_records, PredictionLog::kPredictStreamedLog));
  TF_ASSERT_OK(WriteWarmupData(fname, warmup_records, num_warmup_records));

  auto saved_model = std::make_unique<test_util::MockSavedModel>();

  tfrt::internal::Signature signature;
  signature.input_names = {kPredictInputs};
  signature.input_specs = {tfrt::TensorSpec(tensorflow::DT_STRING)};
  tfrt::FunctionMetadata function_metadata(&signature);
  EXPECT_CALL(*saved_model, GetFunctionMetadata(kPredictMethodName))
      .WillRepeatedly(Return(function_metadata));

  MetaGraphDef meta_graph_def;
  (*meta_graph_def.mutable_signature_def())[kPredictMethodName] =
      CreateSignatureDef(kPredictMethodName, {kPredictInputs}, {});

  EXPECT_CALL(*saved_model, GetMetaGraphDef())
      .WillRepeatedly(ReturnRef(meta_graph_def));

  EXPECT_CALL(
      *saved_model,
      Run(::testing::Field(
              &tfrt_stub::GraphExecutionRunOptions::streamed_output_callback,
              ::testing::NotNull()),
          ::testing::Eq(kPredictMethodName),
          ::testing::An<absl::Span<const Tensor>>(), _))
      .Times(num_warmup_records * GetNumRequestIterations())
      .WillRepeatedly(Return(absl::OkStatus()));

  TF_EXPECT_OK(RunSavedModelWarmup(GetModelWarmupOptions(), base_path,
                                   /*lazy_init_threshold=*/0,
                                   /*skip_warmup_requests_if_initialized=*/true,
                                   saved_model.get()));
}

INSTANTIATE_TEST_SUITE_P(WarmupOptions, TFRTSavedModelWarmupOptionsTest,
                         ::testing::Bool());

TEST(TFRTSavedModelWarmupTest, UnsupportedLogType) {
  string base_path = io::JoinPath(testing::TmpDir(), "UnsupportedLogType");
  TF_ASSERT_OK(Env::Default()->RecursivelyCreateDir(
      io::JoinPath(base_path, kSavedModelAssetsExtraDirectory)));
  string fname = io::JoinPath(base_path, kSavedModelAssetsExtraDirectory,
                              internal::WarmupConsts::kRequestsFileName);

  std::vector<string> warmup_records;
  // Add unsupported log type
  PredictionLog prediction_log;
  TF_ASSERT_OK(
      PopulatePredictionLog(&prediction_log, PredictionLog::kSessionRunLog));
  warmup_records.push_back(prediction_log.SerializeAsString());
  TF_ASSERT_OK(WriteWarmupData(fname, warmup_records, 10));

  std::unique_ptr<test_util::MockSavedModel> saved_model(
      (new test_util::MockSavedModel()));
  MetaGraphDef meta_graph_def;
  AddSignatures(&meta_graph_def);
  EXPECT_CALL(*saved_model, GetMetaGraphDef())
      .WillRepeatedly(ReturnRef(meta_graph_def));
  const absl::Status status = RunSavedModelWarmup(
      ModelWarmupOptions(), base_path,
      /*lazy_init_threshold=*/0,
      /*skip_warmup_requests_if_initialized=*/true, saved_model.get());
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(::tensorflow::error::UNIMPLEMENTED, status.code()) << status;
  EXPECT_THAT(status.ToString(),
              ::testing::HasSubstr("Unsupported log_type for warmup"));
}

TEST(TFRTSavedModelWarmupTest, SkipWarmupRequest) {
  string base_path = io::JoinPath(testing::TmpDir(), "SkipWarmupRequest");
  TF_ASSERT_OK(Env::Default()->RecursivelyCreateDir(
      io::JoinPath(base_path, kSavedModelAssetsExtraDirectory)));
  string fname = io::JoinPath(base_path, kSavedModelAssetsExtraDirectory,
                              internal::WarmupConsts::kRequestsFileName);

  int num_warmup_records = 10;
  std::vector<string> warmup_records;
  TF_ASSERT_OK(AddMixedWarmupData(
      &warmup_records, {PredictionLog::kRegressLog, PredictionLog::kClassifyLog,
                        PredictionLog::kPredictLog}));
  TF_ASSERT_OK(WriteWarmupData(fname, warmup_records, num_warmup_records));

  std::unique_ptr<test_util::MockSavedModel> saved_model(
      (new test_util::MockSavedModel()));
  EXPECT_CALL(*saved_model, GetFunctionMetadata(kPredictMethodName)).Times(0);
  EXPECT_CALL(*saved_model, GetFunctionMetadata(kClassifyMethodName)).Times(0);
  EXPECT_CALL(*saved_model, GetFunctionMetadata(kRegressMethodName)).Times(0);
  MetaGraphDef meta_graph_def;
  AddSignatures(&meta_graph_def);
  EXPECT_CALL(*saved_model, GetMetaGraphDef())
      .WillRepeatedly(ReturnRef(meta_graph_def));

  TF_EXPECT_OK(RunSavedModelWarmup(ModelWarmupOptions(), base_path,
                                   /*lazy_init_threshold=*/10,
                                   /*skip_warmup_requests_if_initialized=*/true,
                                   saved_model.get()));
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
