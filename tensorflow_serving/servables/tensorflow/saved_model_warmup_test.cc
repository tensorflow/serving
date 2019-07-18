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
#include "tensorflow/contrib/session_bundle/signature.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow_serving/apis/classification.pb.h"
#include "tensorflow_serving/apis/inference.pb.h"
#include "tensorflow_serving/apis/input.pb.h"
#include "tensorflow_serving/apis/model.pb.h"
#include "tensorflow_serving/apis/predict.pb.h"
#include "tensorflow_serving/apis/prediction_log.pb.h"
#include "tensorflow_serving/apis/regression.pb.h"
#include "tensorflow_serving/core/test_util/mock_session.h"
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

void PopulateInferenceTask(const string& model_name,
                           const string& signature_name,
                           const string& method_name, InferenceTask* task) {
  ModelSpec model_spec;
  model_spec.set_name(model_name);
  model_spec.set_signature_name(signature_name);
  *task->mutable_model_spec() = model_spec;
  task->set_method_name(method_name);
}

void PopulateMultiInferenceRequest(MultiInferenceRequest* request) {
  request->mutable_input()->mutable_example_list()->add_examples();
  PopulateInferenceTask("test_model", kRegressMethodName, kRegressMethodName,
                        request->add_tasks());
  PopulateInferenceTask("test_model", kClassifyMethodName, kClassifyMethodName,
                        request->add_tasks());
}

void PopulatePredictRequest(PredictRequest* request) {
  request->mutable_model_spec()->set_signature_name(kPredictMethodName);
  TensorProto tensor_proto;
  tensor_proto.add_string_val("input_value");
  tensor_proto.set_dtype(tensorflow::DT_STRING);
  tensor_proto.mutable_tensor_shape()->add_dim()->set_size(1);
  (*request->mutable_inputs())[kPredictInputs] = tensor_proto;
}

void PopulateClassificationRequest(ClassificationRequest* request) {
  request->mutable_input()->mutable_example_list()->add_examples();
  request->mutable_model_spec()->set_signature_name(kClassifyMethodName);
}

void PopulateRegressionRequest(RegressionRequest* request) {
  request->mutable_input()->mutable_example_list()->add_examples();
  request->mutable_model_spec()->set_signature_name(kRegressMethodName);
}

void PopulatePredictionLog(PredictionLog* prediction_log,
                           PredictionLog::LogTypeCase log_type) {
  switch (log_type) {
    case PredictionLog::kRegressLog: {
      PopulateRegressionRequest(
          prediction_log->mutable_regress_log()->mutable_request());
    } break;
    case PredictionLog::kClassifyLog: {
      PopulateClassificationRequest(
          prediction_log->mutable_classify_log()->mutable_request());
    } break;
    case PredictionLog::kPredictLog: {
      PopulatePredictRequest(
          prediction_log->mutable_predict_log()->mutable_request());
    } break;
    case PredictionLog::kMultiInferenceLog: {
      PopulateMultiInferenceRequest(
          prediction_log->mutable_multi_inference_log()->mutable_request());
    } break;
    case PredictionLog::kSessionRunLog:
      prediction_log->mutable_session_run_log();
      TF_FALLTHROUGH_INTENDED;
    default:
      return;
  }
}

Status WriteWarmupData(const string& fname,
                       const std::vector<string>& warmup_records,
                       int num_warmup_records) {
  Env* env = Env::Default();
  std::unique_ptr<WritableFile> file;
  TF_RETURN_IF_ERROR(env->NewWritableFile(fname, &file));

  io::RecordWriterOptions options;
  io::RecordWriter writer(file.get(), options);
  for (int i = 0; i < num_warmup_records; ++i) {
    for (const string& warmup_record : warmup_records) {
      TF_RETURN_IF_ERROR(writer.WriteRecord(warmup_record));
    }
  }
  TF_RETURN_IF_ERROR(writer.Flush());
  return Status::OK();
}

Status WriteWarmupDataAsSerializedProtos(
    const string& fname, const std::vector<string>& warmup_records,
    int num_warmup_records) {
  Env* env = Env::Default();
  std::unique_ptr<WritableFile> file;
  TF_RETURN_IF_ERROR(env->NewWritableFile(fname, &file));
  for (int i = 0; i < num_warmup_records; ++i) {
    for (const string& warmup_record : warmup_records) {
      TF_RETURN_IF_ERROR(file->Append(warmup_record));
    }
  }
  TF_RETURN_IF_ERROR(file->Close());
  return Status::OK();
}

void AddMixedWarmupData(std::vector<string>* warmup_records) {
  for (auto& log_type :
       {PredictionLog::kRegressLog, PredictionLog::kClassifyLog,
        PredictionLog::kPredictLog, PredictionLog::kMultiInferenceLog}) {
    PredictionLog prediction_log;
    PopulatePredictionLog(&prediction_log, log_type);
    warmup_records->push_back(prediction_log.SerializeAsString());
  }
}

// Creates a test SignatureDef with the given parameters
SignatureDef CreateSignatureDef(const string& method_name,
                                const std::vector<string>& input_names,
                                const std::vector<string>& output_names) {
  SignatureDef signature_def;
  signature_def.set_method_name(method_name);
  for (const string& input_name : input_names) {
    TensorInfo input;
    input.set_name(input_name);
    (*signature_def.mutable_inputs())[input_name] = input;
  }
  for (const string& output_name : output_names) {
    TensorInfo output;
    output.set_name(output_name);
    (*signature_def.mutable_outputs())[output_name] = output;
  }
  return signature_def;
}

void AddSignatures(MetaGraphDef* meta_graph_def) {
  (*meta_graph_def->mutable_signature_def())[kRegressMethodName] =
      CreateSignatureDef(kRegressMethodName, {kRegressInputs},
                         {kRegressOutputs});
  (*meta_graph_def->mutable_signature_def())[kClassifyMethodName] =
      CreateSignatureDef(kClassifyMethodName, {kClassifyInputs},
                         {kClassifyOutputClasses, kClassifyOutputScores});
  (*meta_graph_def->mutable_signature_def())[kPredictMethodName] =
      CreateSignatureDef(kPredictMethodName, {kPredictInputs},
                         {kPredictOutputs});
}

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
                              WarmupConsts::kRequestsFileName);

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
  // Regress and Predict case
  EXPECT_CALL(*mock, Run(_, _, SizeIs(1), _, _, _))
      .Times(num_warmup_records * 2 * GetNumRequestIterations())
      .WillRepeatedly(DoAll(SetArgPointee<4>(std::vector<Tensor>({scores})),
                            Return(Status::OK())));
  // Classify case
  EXPECT_CALL(*mock, Run(_, _, SizeIs(2), _, _, _))
      .Times(num_warmup_records * GetNumRequestIterations())
      .WillRepeatedly(
          DoAll(SetArgPointee<4>(std::vector<Tensor>({classes, scores})),
                Return(Status::OK())));
  // MultiInference case
  EXPECT_CALL(*mock, Run(_, _, SizeIs(3), _, _, _))
      .Times(num_warmup_records * GetNumRequestIterations())
      .WillRepeatedly(DoAll(
          SetArgPointee<4>(std::vector<Tensor>({classes, scores, scores})),
          Return(Status::OK())));
  TF_EXPECT_OK(RunSavedModelWarmup(GetModelWarmupOptions(), RunOptions(),
                                   base_path, &saved_model_bundle));
}
INSTANTIATE_TEST_SUITE_P(WarmupOptions, SavedModelBundleWarmupOptionsTest,
                         ::testing::Bool());

TEST(SavedModelBundleWarmupTest, NoWarmupDataFile) {
  string base_path = io::JoinPath(testing::TmpDir(), "NoWarmupDataFile");
  TF_ASSERT_OK(Env::Default()->RecursivelyCreateDir(
      io::JoinPath(base_path, kSavedModelAssetsExtraDirectory)));

  SavedModelBundle saved_model_bundle;
  AddSignatures(&saved_model_bundle.meta_graph_def);
  MockSession* mock = new MockSession;
  saved_model_bundle.session.reset(mock);
  EXPECT_CALL(*mock, Run(_, _, _, _, _, _)).Times(0);
  TF_EXPECT_OK(RunSavedModelWarmup(ModelWarmupOptions(), RunOptions(),
                                   base_path, &saved_model_bundle));
}

TEST(SavedModelBundleWarmupTest, WarmupDataFileEmpty) {
  string base_path = io::JoinPath(testing::TmpDir(), "WarmupDataFileEmpty");
  TF_ASSERT_OK(Env::Default()->RecursivelyCreateDir(
      io::JoinPath(base_path, kSavedModelAssetsExtraDirectory)));
  string fname = io::JoinPath(base_path, kSavedModelAssetsExtraDirectory,
                              WarmupConsts::kRequestsFileName);

  std::vector<string> warmup_records;
  TF_ASSERT_OK(WriteWarmupData(fname, warmup_records, 0));
  SavedModelBundle saved_model_bundle;
  AddSignatures(&saved_model_bundle.meta_graph_def);
  MockSession* mock = new MockSession;
  saved_model_bundle.session.reset(mock);
  EXPECT_CALL(*mock, Run(_, _, _, _, _, _)).Times(0);
  TF_EXPECT_OK(RunSavedModelWarmup(ModelWarmupOptions(), RunOptions(),
                                   base_path, &saved_model_bundle));
}

TEST(SavedModelBundleWarmupTest, UnsupportedLogType) {
  string base_path = io::JoinPath(testing::TmpDir(), "UnsupportedLogType");
  TF_ASSERT_OK(Env::Default()->RecursivelyCreateDir(
      io::JoinPath(base_path, kSavedModelAssetsExtraDirectory)));
  string fname = io::JoinPath(base_path, kSavedModelAssetsExtraDirectory,
                              WarmupConsts::kRequestsFileName);

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
  EXPECT_CALL(*mock, Run(_, _, _, _, _, _))
      .WillRepeatedly(Return(Status::OK()));
  const Status status = RunSavedModelWarmup(ModelWarmupOptions(), RunOptions(),
                                            base_path, &saved_model_bundle);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(::tensorflow::error::UNIMPLEMENTED, status.code()) << status;
  EXPECT_THAT(status.ToString(),
              ::testing::HasSubstr("Unsupported log_type for warmup"));
}

TEST(SavedModelBundleWarmupTest, UnsupportedFileFormat) {
  string base_path = io::JoinPath(testing::TmpDir(), "UnsupportedFileFormat");
  TF_ASSERT_OK(Env::Default()->RecursivelyCreateDir(
      io::JoinPath(base_path, kSavedModelAssetsExtraDirectory)));
  const string fname = io::JoinPath(base_path, kSavedModelAssetsExtraDirectory,
                                    WarmupConsts::kRequestsFileName);

  std::vector<string> warmup_records;
  // Add unsupported log type
  PredictionLog prediction_log;
  PopulatePredictionLog(&prediction_log, PredictionLog::kSessionRunLog);
  warmup_records.push_back(prediction_log.SerializeAsString());

  TF_ASSERT_OK(WriteWarmupDataAsSerializedProtos(fname, warmup_records, 10));
  SavedModelBundle saved_model_bundle;
  AddSignatures(&saved_model_bundle.meta_graph_def);
  MockSession* mock = new MockSession;
  saved_model_bundle.session.reset(mock);
  EXPECT_CALL(*mock, Run(_, _, _, _, _, _))
      .WillRepeatedly(Return(Status::OK()));
  const Status status = RunSavedModelWarmup(ModelWarmupOptions(), RunOptions(),
                                            base_path, &saved_model_bundle);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(::tensorflow::error::DATA_LOSS, status.code()) << status;
  EXPECT_THAT(status.ToString(),
              ::testing::HasSubstr(
                  "Please verify your warmup data is in TFRecord format"));
}

TEST(SavedModelBundleWarmupTest, TooManyWarmupRecords) {
  string base_path = io::JoinPath(testing::TmpDir(), "TooManyWarmupRecords");
  TF_ASSERT_OK(Env::Default()->RecursivelyCreateDir(
      io::JoinPath(base_path, kSavedModelAssetsExtraDirectory)));
  string fname = io::JoinPath(base_path, kSavedModelAssetsExtraDirectory,
                              WarmupConsts::kRequestsFileName);

  std::vector<string> warmup_records;
  AddMixedWarmupData(&warmup_records);
  TF_ASSERT_OK(
      WriteWarmupData(fname, warmup_records, WarmupConsts::kMaxNumRecords + 1));
  SavedModelBundle saved_model_bundle;
  AddSignatures(&saved_model_bundle.meta_graph_def);
  MockSession* mock = new MockSession;
  saved_model_bundle.session.reset(mock);
  Tensor scores(DT_FLOAT, TensorShape({1, 1}));
  Tensor classes(DT_STRING, TensorShape({1, 1}));
  // Regress and Predict case
  EXPECT_CALL(*mock, Run(_, _, SizeIs(1), _, _, _))
      .WillRepeatedly(DoAll(SetArgPointee<4>(std::vector<Tensor>({scores})),
                            Return(Status::OK())));
  // Classify case
  EXPECT_CALL(*mock, Run(_, _, SizeIs(2), _, _, _))
      .WillRepeatedly(
          DoAll(SetArgPointee<4>(std::vector<Tensor>({classes, scores})),
                Return(Status::OK())));
  // MultiInference case
  EXPECT_CALL(*mock, Run(_, _, SizeIs(3), _, _, _))
      .WillRepeatedly(DoAll(
          SetArgPointee<4>(std::vector<Tensor>({classes, scores, scores})),
          Return(Status::OK())));
  const Status status = RunSavedModelWarmup(ModelWarmupOptions(), RunOptions(),
                                            base_path, &saved_model_bundle);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, status.code()) << status;
  EXPECT_THAT(
      status.ToString(),
      ::testing::HasSubstr("Number of warmup records exceeeds the maximum"));
}

TEST(SavedModelBundleWarmupTest, UnparsableRecord) {
  string base_path = io::JoinPath(testing::TmpDir(), "UnparsableRecord");
  TF_ASSERT_OK(Env::Default()->RecursivelyCreateDir(
      io::JoinPath(base_path, kSavedModelAssetsExtraDirectory)));
  string fname = io::JoinPath(base_path, kSavedModelAssetsExtraDirectory,
                              WarmupConsts::kRequestsFileName);

  std::vector<string> warmup_records = {"malformed_record"};
  TF_ASSERT_OK(WriteWarmupData(fname, warmup_records, 10));
  SavedModelBundle saved_model_bundle;
  MockSession* mock = new MockSession;
  saved_model_bundle.session.reset(mock);
  EXPECT_CALL(*mock, Run(_, _, _, _, _, _)).Times(0);
  const Status status = RunSavedModelWarmup(ModelWarmupOptions(), RunOptions(),
                                            base_path, &saved_model_bundle);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, status.code()) << status;
  EXPECT_THAT(status.ToString(),
              ::testing::HasSubstr("Failed to parse warmup record"));
}

TEST(SavedModelBundleWarmupTest, RunFailure) {
  string base_path = io::JoinPath(testing::TmpDir(), "RunFailure");
  TF_ASSERT_OK(Env::Default()->RecursivelyCreateDir(
      io::JoinPath(base_path, kSavedModelAssetsExtraDirectory)));
  string fname = io::JoinPath(base_path, kSavedModelAssetsExtraDirectory,
                              WarmupConsts::kRequestsFileName);

  int num_warmup_records = 10;
  std::vector<string> warmup_records;
  AddMixedWarmupData(&warmup_records);
  TF_ASSERT_OK(WriteWarmupData(fname, warmup_records, num_warmup_records));
  SavedModelBundle saved_model_bundle;
  AddSignatures(&saved_model_bundle.meta_graph_def);
  MockSession* mock = new MockSession;
  saved_model_bundle.session.reset(mock);
  EXPECT_CALL(*mock, Run(_, _, _, _, _, _))
      .WillOnce(::testing::Return(errors::InvalidArgument("Run failed")));
  const Status status = RunSavedModelWarmup(ModelWarmupOptions(), RunOptions(),
                                            base_path, &saved_model_bundle);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, status.code()) << status;
  EXPECT_THAT(status.ToString(), ::testing::HasSubstr("Run failed"));
}

}  // namespace

}  // namespace serving
}  // namespace tensorflow
