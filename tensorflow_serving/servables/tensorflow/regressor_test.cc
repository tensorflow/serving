/* Copyright 2017 Google Inc. All Rights Reserved.

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

#include "tensorflow_serving/servables/tensorflow/regressor.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/map.h"
#include "absl/types/optional.h"
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/threadpool_options.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow_serving/apis/input.pb.h"
#include "tensorflow_serving/apis/model.pb.h"
#include "tensorflow_serving/apis/regression.pb.h"
#include "tensorflow_serving/core/test_util/mock_session.h"
#include "tensorflow_serving/servables/tensorflow/util.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace {

using test_util::EqualsProto;
using test_util::MockSession;
using ::testing::_;

const char kInputTensor[] = "input:0";
const char kOutputTensor[] = "output:0";
const char kOutputPlusOneTensor[] = "outputPlusOne:0";
const char kImproperlySizedOutputTensor[] = "ImproperlySizedOutput:0";
const char kOutputFeature[] = "output";

const char kOutputPlusOneSignature[] = "output_plus_one";
const char kInvalidNamedSignature[] = "invalid_classification_signature";
const char kImproperlySizedOutputSignature[] = "ImproperlySizedOutputSignature";

// Fake Session used for testing TensorFlowRegressor
// Assumes the input Tensor "input:0" has serialized tensorflow::Example values.
// Copies the "output" float feature from each Example.
class FakeSession : public tensorflow::Session {
 public:
  explicit FakeSession(absl::optional<int64_t> expected_timeout)
      : expected_timeout_(expected_timeout) {}
  ~FakeSession() override = default;
  Status Create(const GraphDef& graph) override {
    return errors::Unimplemented("not available in fake");
  }
  Status Extend(const GraphDef& graph) override {
    return errors::Unimplemented("not available in fake");
  }

  Status Close() override {
    return errors::Unimplemented("not available in fake");
  }

  Status ListDevices(std::vector<DeviceAttributes>* response) override {
    return errors::Unimplemented("not available in fake");
  }

  Status Run(const std::vector<std::pair<string, Tensor>>& inputs,
             const std::vector<string>& output_names,
             const std::vector<string>& target_nodes,
             std::vector<Tensor>* outputs) override {
    if (expected_timeout_) {
      LOG(FATAL) << "Run() without RunOptions not expected to be called";
    }
    RunMetadata run_metadata;
    return Run(RunOptions(), inputs, output_names, target_nodes, outputs,
               &run_metadata);
  }

  Status Run(const RunOptions& run_options,
             const std::vector<std::pair<string, Tensor>>& inputs,
             const std::vector<string>& output_names,
             const std::vector<string>& target_nodes,
             std::vector<Tensor>* outputs, RunMetadata* run_metadata) override {
    return Run(run_options, inputs, output_names, target_nodes, outputs,
               run_metadata, thread::ThreadPoolOptions());
  }

  Status Run(const RunOptions& run_options,
             const std::vector<std::pair<string, Tensor>>& inputs,
             const std::vector<string>& output_names,
             const std::vector<string>& target_nodes,
             std::vector<Tensor>* outputs, RunMetadata* run_metadata,
             const thread::ThreadPoolOptions& thread_pool_options) override {
    if (expected_timeout_) {
      CHECK_EQ(*expected_timeout_, run_options.timeout_in_ms());
    }
    if (inputs.size() != 1 || inputs[0].first != kInputTensor) {
      return errors::Internal("Expected one input Tensor.");
    }
    const Tensor& input = inputs[0].second;
    std::vector<Example> examples;
    TF_RETURN_IF_ERROR(GetExamples(input, &examples));
    Tensor output;
    TF_RETURN_IF_ERROR(GetOutputTensor(examples, output_names[0], &output));
    outputs->push_back(output);
    return OkStatus();
  }

  // Parses TensorFlow Examples from a string Tensor.
  static Status GetExamples(const Tensor& input,
                            std::vector<Example>* examples) {
    examples->clear();
    const int batch_size = input.dim_size(0);
    const auto& flat_input = input.flat<tstring>();
    for (int i = 0; i < batch_size; ++i) {
      Example example;
      if (!example.ParseFromArray(flat_input(i).data(), flat_input(i).size())) {
        return errors::Internal("failed to parse example");
      }
      examples->push_back(example);
    }
    return OkStatus();
  }

  // Gets the Feature from an Example with the given name.  Returns empty
  // Feature if the name does not exist.
  static Feature GetFeature(const Example& example, const string& name) {
    const auto it = example.features().feature().find(name);
    if (it != example.features().feature().end()) {
      return it->second;
    }
    return Feature();
  }

  // Creates a Tensor by copying the "output" feature from each Example.
  // Requires each Example have an bytes feature called "class" which is of the
  // same non-zero length.
  static Status GetOutputTensor(const std::vector<Example>& examples,
                                const string& output_tensor_name,
                                Tensor* tensor) {
    if (examples.empty()) {
      return errors::Internal("empty example list");
    }
    const int batch_size = examples.size();
    if (output_tensor_name == kImproperlySizedOutputTensor) {
      // Insert a rank 3 tensor which should be an error because outputs are
      // expected to be of shape [batch_size] or [batch_size, 1].
      *tensor = Tensor(DT_FLOAT, TensorShape({batch_size, 1, 10}));
      return OkStatus();
    }
    // Both tensor shapes are valid, so make one of shape [batch_size, 1] and
    // the rest of shape [batch_size].
    *tensor = output_tensor_name == kOutputPlusOneTensor
                  ? Tensor(DT_FLOAT, TensorShape({batch_size, 1}))
                  : Tensor(DT_FLOAT, TensorShape({batch_size}));

    const float offset = output_tensor_name == kOutputPlusOneTensor ? 1 : 0;
    for (int i = 0; i < batch_size; ++i) {
      const Feature feature = GetFeature(examples[i], kOutputFeature);
      if (feature.float_list().value_size() != 1) {
        return errors::Internal("incorrect number of values in output feature");
      }
      tensor->flat<float>()(i) = feature.float_list().value(0) + offset;
    }
    return OkStatus();
  }

 private:
  const absl::optional<int64_t> expected_timeout_;
};

class RegressorTest : public ::testing::TestWithParam<bool> {
 public:
  void SetUp() override {
    SetSignatureMethodNameCheckFeature(IsMethodNameCheckEnabled());
    saved_model_bundle_.reset(new SavedModelBundle);
    meta_graph_def_ = &saved_model_bundle_->meta_graph_def;
    absl::optional<int64_t> expected_timeout = GetRunOptions().timeout_in_ms();
    fake_session_ = new FakeSession(expected_timeout);
    saved_model_bundle_->session.reset(fake_session_);

    auto* signature_defs = meta_graph_def_->mutable_signature_def();
    SignatureDef sig_def;
    TensorInfo input_tensor_info;
    input_tensor_info.set_name(kInputTensor);
    (*sig_def.mutable_inputs())[kRegressInputs] = input_tensor_info;
    TensorInfo scores_tensor_info;
    scores_tensor_info.set_name(kOutputTensor);
    (*sig_def.mutable_outputs())[kRegressOutputs] = scores_tensor_info;
    if (IsMethodNameCheckEnabled()) sig_def.set_method_name(kRegressMethodName);
    (*signature_defs)[kDefaultServingSignatureDefKey] = sig_def;

    AddNamedSignatureToSavedModelBundle(
        kInputTensor, kOutputPlusOneTensor, kOutputPlusOneSignature,
        true /* is_regression */, meta_graph_def_);
    AddNamedSignatureToSavedModelBundle(
        kInputTensor, kOutputPlusOneTensor, kInvalidNamedSignature,
        false /* is_regression */, meta_graph_def_);

    // Add a named signature where the output is not valid.
    AddNamedSignatureToSavedModelBundle(
        kInputTensor, kImproperlySizedOutputTensor,
        kImproperlySizedOutputSignature, true /* is_regression */,
        meta_graph_def_);
  }

 protected:
  bool IsMethodNameCheckEnabled() { return GetParam(); }

  // Return an example with the feature "output" = [output].
  Example example_with_output(const float output) {
    Feature feature;
    feature.mutable_float_list()->add_value(output);
    Example example;
    (*example.mutable_features()->mutable_feature())["output"] = feature;
    return example;
  }

  Status Create() {
    std::unique_ptr<SavedModelBundle> saved_model(new SavedModelBundle);
    saved_model->meta_graph_def = saved_model_bundle_->meta_graph_def;
    saved_model->session = std::move(saved_model_bundle_->session);
    return CreateRegressorFromSavedModelBundle(
        GetRunOptions(), std::move(saved_model), &regressor_);
  }

  RunOptions GetRunOptions() const {
    RunOptions run_options;
    run_options.set_timeout_in_ms(42);
    return run_options;
  }

  // Add a named signature to the mutable meta_graph_def* parameter.
  // If is_regression is false, will add a classification signature, which is
  // invalid in classification requests.
  void AddNamedSignatureToSavedModelBundle(
      const string& input_tensor_name, const string& output_scores_tensor_name,
      const string& signature_name, const bool is_regression,
      tensorflow::MetaGraphDef* meta_graph_def) {
    auto* signature_defs = meta_graph_def->mutable_signature_def();
    SignatureDef sig_def;
    string method_name;
    if (is_regression) {
      TensorInfo input_tensor_info;
      input_tensor_info.set_name(input_tensor_name);
      (*sig_def.mutable_inputs())[kRegressInputs] = input_tensor_info;
      TensorInfo scores_tensor_info;
      scores_tensor_info.set_name(output_scores_tensor_name);
      (*sig_def.mutable_outputs())[kRegressOutputs] = scores_tensor_info;
      method_name = kRegressMethodName;
    } else {
      TensorInfo input_tensor_info;
      input_tensor_info.set_name(input_tensor_name);
      (*sig_def.mutable_inputs())[kClassifyInputs] = input_tensor_info;
      TensorInfo class_tensor_info;
      class_tensor_info.set_name(kOutputPlusOneTensor);
      (*sig_def.mutable_outputs())[kClassifyOutputClasses] = class_tensor_info;
      method_name = kClassifyMethodName;
    }
    if (IsMethodNameCheckEnabled()) sig_def.set_method_name(method_name);
    (*signature_defs)[signature_name] = sig_def;
  }

  // Variables used to create the regression model
  tensorflow::MetaGraphDef* meta_graph_def_;
  FakeSession* fake_session_;
  std::unique_ptr<SavedModelBundle> saved_model_bundle_;

  // Regression model valid after calling create.
  std::unique_ptr<RegressorInterface> regressor_;

  // Convenience variables.
  RegressionRequest request_;
  RegressionResult result_;
};

TEST_P(RegressorTest, BasicExampleList) {
  TF_ASSERT_OK(Create());
  auto* examples =
      request_.mutable_input()->mutable_example_list()->mutable_examples();
  *examples->Add() = example_with_output(2.0);
  *examples->Add() = example_with_output(3.0);
  TF_ASSERT_OK(regressor_->Regress(request_, &result_));
  EXPECT_THAT(result_, EqualsProto(" regressions { "
                                   "   value: 2.0 "
                                   " } "
                                   " regressions { "
                                   "   value: 3.0 "
                                   " } "));
  RegressionResponse response;
  TF_ASSERT_OK(RunRegress(GetRunOptions(), saved_model_bundle_->meta_graph_def,
                          {}, fake_session_, request_, &response));
  EXPECT_THAT(response.result(), EqualsProto(" regressions { "
                                             "   value: 2.0 "
                                             " } "
                                             " regressions { "
                                             "   value: 3.0 "
                                             " } "));
}

TEST_P(RegressorTest, BasicExampleListWithContext) {
  TF_ASSERT_OK(Create());
  auto* list_with_context =
      request_.mutable_input()->mutable_example_list_with_context();
  // Add two empty examples.
  list_with_context->add_examples();
  list_with_context->add_examples();
  // Add the context which contains the output predictions.
  *list_with_context->mutable_context() = example_with_output(3.0);
  TF_ASSERT_OK(regressor_->Regress(request_, &result_));
  EXPECT_THAT(result_, EqualsProto(" regressions { "
                                   "   value: 3.0 "
                                   " } "
                                   " regressions { "
                                   "   value: 3.0 "
                                   " } "));
  RegressionResponse response;
  TF_ASSERT_OK(RunRegress(GetRunOptions(), saved_model_bundle_->meta_graph_def,
                          {}, fake_session_, request_, &response));
  EXPECT_THAT(response.result(), EqualsProto(" regressions { "
                                             "   value: 3.0 "
                                             " } "
                                             " regressions { "
                                             "   value: 3.0 "
                                             " } "));
}

TEST_P(RegressorTest, ValidNamedSignature) {
  TF_ASSERT_OK(Create());
  request_.mutable_model_spec()->set_signature_name(kOutputPlusOneSignature);
  auto* examples =
      request_.mutable_input()->mutable_example_list()->mutable_examples();
  *examples->Add() = example_with_output(2.0);
  *examples->Add() = example_with_output(3.0);
  TF_ASSERT_OK(regressor_->Regress(request_, &result_));
  EXPECT_THAT(result_, EqualsProto(" regressions { "
                                   "   value: 3.0 "
                                   " } "
                                   " regressions { "
                                   "   value: 4.0 "
                                   " } "));

  RegressionResponse response;
  TF_ASSERT_OK(RunRegress(GetRunOptions(), saved_model_bundle_->meta_graph_def,
                          {}, fake_session_, request_, &response));
  EXPECT_THAT(response.result(), EqualsProto(" regressions { "
                                             "   value: 3.0 "
                                             " } "
                                             " regressions { "
                                             "   value: 4.0 "
                                             " } "));
}

TEST_P(RegressorTest, InvalidNamedSignature) {
  TF_ASSERT_OK(Create());
  request_.mutable_model_spec()->set_signature_name(kInvalidNamedSignature);
  auto* examples =
      request_.mutable_input()->mutable_example_list()->mutable_examples();
  *examples->Add() = example_with_output(2.0);
  *examples->Add() = example_with_output(3.0);
  Status status = regressor_->Regress(request_, &result_);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, status.code()) << status;

  RegressionResponse response;
  status = RunRegress(GetRunOptions(), saved_model_bundle_->meta_graph_def, {},
                      fake_session_, request_, &response);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, status.code()) << status;
}

TEST_P(RegressorTest, MalformedOutputs) {
  TF_ASSERT_OK(Create());
  request_.mutable_model_spec()->set_signature_name(
      kImproperlySizedOutputSignature);
  auto* examples =
      request_.mutable_input()->mutable_example_list()->mutable_examples();
  *examples->Add() = example_with_output(2.0);
  *examples->Add() = example_with_output(3.0);
  Status status = regressor_->Regress(request_, &result_);

  ASSERT_FALSE(status.ok());
  EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, status.code()) << status;
  // Test RunRegress
  RegressionResponse response;
  status = RunRegress(GetRunOptions(), saved_model_bundle_->meta_graph_def, {},
                      fake_session_, request_, &response);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, status.code()) << status;
}

TEST_P(RegressorTest, EmptyInput) {
  TF_ASSERT_OK(Create());
  // Touch input.
  request_.mutable_input();
  Status status = regressor_->Regress(request_, &result_);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.code(), error::Code::INVALID_ARGUMENT);
  EXPECT_THAT(status.error_message(), ::testing::HasSubstr("Input is empty"));
  RegressionResponse response;
  status = RunRegress(GetRunOptions(), saved_model_bundle_->meta_graph_def, {},
                      fake_session_, request_, &response);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.code(), error::Code::INVALID_ARGUMENT);
  EXPECT_THAT(status.error_message(), ::testing::HasSubstr("Input is empty"));
}

TEST_P(RegressorTest, EmptyExampleList) {
  TF_ASSERT_OK(Create());
  request_.mutable_input()->mutable_example_list();
  Status status = regressor_->Regress(request_, &result_);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.code(), error::Code::INVALID_ARGUMENT);
  EXPECT_THAT(status.error_message(), ::testing::HasSubstr("Input is empty"));
  RegressionResponse response;
  status = RunRegress(GetRunOptions(), saved_model_bundle_->meta_graph_def, {},
                      fake_session_, request_, &response);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.code(), error::Code::INVALID_ARGUMENT);
  EXPECT_THAT(status.error_message(), ::testing::HasSubstr("Input is empty"));
}

TEST_P(RegressorTest, EmptyExampleListWithContext) {
  TF_ASSERT_OK(Create());
  // Add a ExampleListWithContext which has context but no examples.
  *request_.mutable_input()
       ->mutable_example_list_with_context()
       ->mutable_context() = example_with_output(3);
  Status status = regressor_->Regress(request_, &result_);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.code(), error::Code::INVALID_ARGUMENT);
  EXPECT_THAT(status.error_message(), ::testing::HasSubstr("Input is empty"));
  RegressionResponse response;
  status = RunRegress(GetRunOptions(), saved_model_bundle_->meta_graph_def, {},
                      fake_session_, request_, &response);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.code(), error::Code::INVALID_ARGUMENT);
  EXPECT_THAT(status.error_message(), ::testing::HasSubstr("Input is empty"));
}

TEST_P(RegressorTest, RunsFails) {
  MockSession* mock = new MockSession;
  saved_model_bundle_->session.reset(mock);
  EXPECT_CALL(*mock, Run(_, _, _, _, _, _, _))
      .WillRepeatedly(
          ::testing::Return(errors::Internal("Run totally failed")));
  TF_ASSERT_OK(Create());
  *request_.mutable_input()->mutable_example_list()->mutable_examples()->Add() =
      example_with_output(2.0);
  Status status = regressor_->Regress(request_, &result_);
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), ::testing::HasSubstr("Run totally failed"));
  RegressionResponse response;
  status = RunRegress(GetRunOptions(), saved_model_bundle_->meta_graph_def, {},
                      mock, request_, &response);
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), ::testing::HasSubstr("Run totally failed"));
}

TEST_P(RegressorTest, UnexpectedOutputTensorSize) {
  MockSession* mock = new MockSession;
  saved_model_bundle_->session.reset(mock);
  std::vector<Tensor> outputs = {Tensor(DT_FLOAT, TensorShape({2}))};
  EXPECT_CALL(*mock, Run(_, _, _, _, _, _, _))
      .WillOnce(::testing::DoAll(::testing::SetArgPointee<4>(outputs),
                                 ::testing::Return(OkStatus())));
  TF_ASSERT_OK(Create());
  *request_.mutable_input()->mutable_example_list()->mutable_examples()->Add() =
      example_with_output(2.0);
  Status status = regressor_->Regress(request_, &result_);
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), ::testing::HasSubstr("output batch size"));
  EXPECT_CALL(*mock, Run(_, _, _, _, _, _, _))
      .WillOnce(::testing::DoAll(::testing::SetArgPointee<4>(outputs),
                                 ::testing::Return(OkStatus())));
  RegressionResponse response;
  status = RunRegress(GetRunOptions(), saved_model_bundle_->meta_graph_def, {},
                      mock, request_, &response);
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), ::testing::HasSubstr("output batch size"));
}

TEST_P(RegressorTest, UnexpectedOutputTensorType) {
  MockSession* mock = new MockSession;
  saved_model_bundle_->session.reset(mock);
  // We expect a FLOAT output type; test returning a STRING.
  std::vector<Tensor> outputs = {Tensor(DT_STRING, TensorShape({1}))};
  EXPECT_CALL(*mock, Run(_, _, _, _, _, _, _))
      .WillOnce(::testing::DoAll(::testing::SetArgPointee<4>(outputs),
                                 ::testing::Return(OkStatus())));
  TF_ASSERT_OK(Create());
  *request_.mutable_input()->mutable_example_list()->mutable_examples()->Add() =
      example_with_output(2.0);
  Status status = regressor_->Regress(request_, &result_);
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(),
              ::testing::HasSubstr("Expected output Tensor of DT_FLOAT"));
  EXPECT_CALL(*mock, Run(_, _, _, _, _, _, _))
      .WillOnce(::testing::DoAll(::testing::SetArgPointee<4>(outputs),
                                 ::testing::Return(OkStatus())));
  RegressionResponse response;
  status = RunRegress(GetRunOptions(), saved_model_bundle_->meta_graph_def, {},
                      mock, request_, &response);
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(),
              ::testing::HasSubstr("Expected output Tensor of DT_FLOAT"));
}

TEST_P(RegressorTest, MissingRegressionSignature) {
  auto* signature_defs = meta_graph_def_->mutable_signature_def();
  SignatureDef sig_def;
  (*signature_defs)[kDefaultServingSignatureDefKey] = sig_def;
  TF_ASSERT_OK(Create());
  Feature feature;
  feature.mutable_bytes_list()->add_value("uno");
  Example example;
  (*example.mutable_features()->mutable_feature())["class"] = feature;
  *request_.mutable_input()->mutable_example_list()->mutable_examples()->Add() =
      example;
  // TODO(b/26220896): This error should move to construction time.
  Status status = regressor_->Regress(request_, &result_);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, status.code()) << status;
  RegressionResponse response;
  status = RunRegress(GetRunOptions(), saved_model_bundle_->meta_graph_def, {},
                      fake_session_, request_, &response);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, status.code()) << status;
}

TEST_P(RegressorTest, MethodNameCheck) {
  RegressionResponse response;
  *request_.mutable_input()->mutable_example_list()->mutable_examples()->Add() =
      example_with_output(2.0);
  auto* signature_defs = meta_graph_def_->mutable_signature_def();

  // Legit method name. Should always work.
  (*signature_defs)[kDefaultServingSignatureDefKey].set_method_name(
      kRegressMethodName);
  TF_EXPECT_OK(RunRegress(GetRunOptions(), *meta_graph_def_, {}, fake_session_,
                          request_, &response));

  // Unsupported method name will fail when method check is enabled.
  (*signature_defs)[kDefaultServingSignatureDefKey].set_method_name(
      "not/supported/method");
  EXPECT_EQ(RunRegress(GetRunOptions(), *meta_graph_def_, {}, fake_session_,
                       request_, &response)
                .ok(),
            !IsMethodNameCheckEnabled());

  // Empty method name will fail when method check is enabled.
  (*signature_defs)[kDefaultServingSignatureDefKey].clear_method_name();
  EXPECT_EQ(RunRegress(GetRunOptions(), *meta_graph_def_, {}, fake_session_,
                       request_, &response)
                .ok(),
            !IsMethodNameCheckEnabled());
}

INSTANTIATE_TEST_SUITE_P(Regressor, RegressorTest, ::testing::Bool());

}  // namespace
}  // namespace serving
}  // namespace tensorflow
