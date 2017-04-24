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
#include "tensorflow/contrib/session_bundle/bundle_shim.h"
#include "tensorflow/contrib/session_bundle/session_bundle.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow_serving/apis/input.pb.h"
#include "tensorflow_serving/apis/model.pb.h"
#include "tensorflow_serving/apis/regression.pb.h"
#include "tensorflow_serving/core/test_util/mock_session.h"
#include "tensorflow_serving/test_util/test_util.h"
#include "tensorflow_serving/util/optional.h"

namespace tensorflow {
namespace serving {
namespace {

using ::testing::_;
using test_util::EqualsProto;
using test_util::MockSession;

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
  explicit FakeSession(optional<int64> expected_timeout)
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
    return Status::OK();
  }

  // Parses TensorFlow Examples from a string Tensor.
  static Status GetExamples(const Tensor& input,
                            std::vector<Example>* examples) {
    examples->clear();
    const int batch_size = input.dim_size(0);
    const auto& flat_input = input.flat<string>();
    for (int i = 0; i < batch_size; ++i) {
      Example example;
      if (!example.ParseFromString(flat_input(i))) {
        return errors::Internal("failed to parse example");
      }
      examples->push_back(example);
    }
    return Status::OK();
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
      return Status::OK();
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
    return Status::OK();
  }

 private:
  const optional<int64> expected_timeout_;
};

// Add a named signature to the mutable signatures* parameter.
// If is_regression is false, will add a classification signature, which is
// invalid in regression requests.
void AddNamedSignature(const string& input_tensor_name,
                       const string& output_tensor_name,
                       const string& signature_name, const bool is_regression,
                       Signatures* signatures) {
  tensorflow::serving::Signature named_signature;
  if (is_regression) {
    named_signature.mutable_regression_signature()
        ->mutable_input()
        ->set_tensor_name(input_tensor_name);
    named_signature.mutable_regression_signature()
        ->mutable_output()
        ->set_tensor_name(output_tensor_name);
  } else {
    named_signature.mutable_classification_signature()
        ->mutable_input()
        ->set_tensor_name(input_tensor_name);
    named_signature.mutable_classification_signature()
        ->mutable_classes()
        ->set_tensor_name(output_tensor_name);
  }
  signatures->mutable_named_signatures()->insert(
      protobuf::MapPair<string, tensorflow::serving::Signature>(
          signature_name, named_signature));
}

// Parameter is 'bool use_saved_model'.
class RegressorTest : public ::testing::TestWithParam<bool> {
 public:
  void SetUp() override {
    bundle_.reset(new SessionBundle);
    meta_graph_def_ = &bundle_->meta_graph_def;
    optional<int64> expected_timeout = GetRunOptions().timeout_in_ms();
    if (!GetParam()) {
      // For SessionBundle we don't propagate the timeout.
      expected_timeout = nullopt;
    }
    fake_session_ = new FakeSession(expected_timeout);
    bundle_->session.reset(fake_session_);

    // Setup some defaults for our signature.
    tensorflow::serving::Signatures signatures;
    auto default_signature =
        signatures.mutable_default_signature()->mutable_regression_signature();
    default_signature->mutable_input()->set_tensor_name(kInputTensor);
    default_signature->mutable_output()->set_tensor_name(kOutputTensor);

    AddNamedSignature(kInputTensor, kOutputPlusOneTensor,
                      kOutputPlusOneSignature, true /* is_regression */,
                      &signatures);
    AddNamedSignature(kInputTensor, kOutputPlusOneTensor,
                      kInvalidNamedSignature, false /* is_regression */,
                      &signatures);

    // Add a named signature where the output is not valid.
    AddNamedSignature(kInputTensor, kImproperlySizedOutputTensor,
                      kImproperlySizedOutputSignature, true /* is_regression */,
                      &signatures);

    TF_ASSERT_OK(
        tensorflow::serving::SetSignatures(signatures, meta_graph_def_));
  }

 protected:
  // Return an example with the feature "output" = [output].
  Example example_with_output(const float output) {
    Feature feature;
    feature.mutable_float_list()->add_value(output);
    Example example;
    (*example.mutable_features()->mutable_feature())["output"] = feature;
    return example;
  }

  Status Create() {
    if (GetParam()) {
      std::unique_ptr<SavedModelBundle> saved_model(new SavedModelBundle);
      TF_CHECK_OK(internal::ConvertSessionBundleToSavedModelBundle(
          *bundle_, saved_model.get()));
      return CreateRegressorFromSavedModelBundle(
          GetRunOptions(), std::move(saved_model), &regressor_);
    } else {
      return CreateRegressorFromBundle(std::move(bundle_), &regressor_);
    }
  }

  // Variables used to create the regression model
  tensorflow::MetaGraphDef* meta_graph_def_;
  FakeSession* fake_session_;
  std::unique_ptr<SessionBundle> bundle_;

  // Regression model valid after calling create.
  std::unique_ptr<RegressorInterface> regressor_;

  // Convenience variables.
  RegressionRequest request_;
  RegressionResult result_;

 private:
  RunOptions GetRunOptions() const {
    RunOptions run_options;
    run_options.set_timeout_in_ms(42);
    return run_options;
  }
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
}

TEST_P(RegressorTest, ValidNamedSignature) {
  TF_ASSERT_OK(Create());
  request_.mutable_model_spec()->set_signature_name(kOutputPlusOneSignature);
  auto* examples =
      request_.mutable_input()->mutable_example_list()->mutable_examples();
  *examples->Add() = example_with_output(2.0);
  *examples->Add() = example_with_output(3.0);
  TF_ASSERT_OK(regressor_->Regress(request_, &result_));
  // GetParam() is 'is_saved_model' in this test. If using saved_model, this
  // test should use the kOutputPlusOneSignature named signature. Otherwise,
  // when using session_bundle, the signature_name in the model_spec will be
  // ignored and the default signature will be used.
  if (GetParam()) {
    EXPECT_THAT(result_, EqualsProto(" regressions { "
                                     "   value: 3.0 "
                                     " } "
                                     " regressions { "
                                     "   value: 4.0 "
                                     " } "));
  } else {
    EXPECT_THAT(result_, EqualsProto(" regressions { "
                                     "   value: 2.0 "
                                     " } "
                                     " regressions { "
                                     "   value: 3.0 "
                                     " } "));
  }
}

TEST_P(RegressorTest, InvalidNamedSignature) {
  TF_ASSERT_OK(Create());
  request_.mutable_model_spec()->set_signature_name(kInvalidNamedSignature);
  auto* examples =
      request_.mutable_input()->mutable_example_list()->mutable_examples();
  *examples->Add() = example_with_output(2.0);
  *examples->Add() = example_with_output(3.0);
  const Status status = regressor_->Regress(request_, &result_);

  // GetParam() is 'is_saved_model' in this test. If using saved_model, this
  // test should fail because the named_signature requested is actually a
  // classification signature. When using session_bundle, the signature_name
  // will be ignored and the default signature will be used.
  if (GetParam()) {
    ASSERT_FALSE(status.ok());
    EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, status.code()) << status;
  } else {
    TF_ASSERT_OK(status);
    EXPECT_THAT(result_, EqualsProto(" regressions { "
                                     "   value: 2.0 "
                                     " } "
                                     " regressions { "
                                     "   value: 3.0 "
                                     " } "));
  }
}

TEST_P(RegressorTest, MalformedOutputs) {
  // If not using SavedModel, we don't use named signatures so the test is not
  // actually testing the right thing. Skip it.
  if (!GetParam()) {
    return;
  }

  TF_ASSERT_OK(Create());
  request_.mutable_model_spec()->set_signature_name(
      kImproperlySizedOutputSignature);
  auto* examples =
      request_.mutable_input()->mutable_example_list()->mutable_examples();
  *examples->Add() = example_with_output(2.0);
  *examples->Add() = example_with_output(3.0);
  const Status status = regressor_->Regress(request_, &result_);

  ASSERT_FALSE(status.ok());
  EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, status.code()) << status;
}

TEST_P(RegressorTest, EmptyInput) {
  TF_ASSERT_OK(Create());
  // Touch input.
  request_.mutable_input();
  const Status status = regressor_->Regress(request_, &result_);
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(),
              ::testing::HasSubstr("Invalid argument: Input is empty"));
}

TEST_P(RegressorTest, EmptyExampleList) {
  TF_ASSERT_OK(Create());
  request_.mutable_input()->mutable_example_list();
  const Status status = regressor_->Regress(request_, &result_);
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(),
              ::testing::HasSubstr("Invalid argument: Input is empty"));
}

TEST_P(RegressorTest, EmptyExampleListWithContext) {
  TF_ASSERT_OK(Create());
  // Add a ExampleListWithContext which has context but no examples.
  *request_.mutable_input()
       ->mutable_example_list_with_context()
       ->mutable_context() = example_with_output(3);
  const Status status = regressor_->Regress(request_, &result_);
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(),
              ::testing::HasSubstr("Invalid argument: Input is empty"));
}

TEST_P(RegressorTest, RunsFails) {
  MockSession* mock = new MockSession;
  bundle_->session.reset(mock);
  if (GetParam()) {
    EXPECT_CALL(*mock, Run(_, _, _, _, _, _))
        .WillRepeatedly(
            ::testing::Return(errors::Internal("Run totally failed")));
  } else {
    EXPECT_CALL(*mock, Run(_, _, _, _))
        .WillRepeatedly(
            ::testing::Return(errors::Internal("Run totally failed")));
  }
  TF_ASSERT_OK(Create());
  *request_.mutable_input()->mutable_example_list()->mutable_examples()->Add() =
      example_with_output(2.0);
  const Status status = regressor_->Regress(request_, &result_);
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), ::testing::HasSubstr("Run totally failed"));
}

TEST_P(RegressorTest, UnexpectedOutputTensorSize) {
  MockSession* mock = new MockSession;
  bundle_->session.reset(mock);
  std::vector<Tensor> outputs = {Tensor(DT_FLOAT, TensorShape({2}))};
  if (GetParam()) {
    EXPECT_CALL(*mock, Run(_, _, _, _, _, _))
        .WillOnce(::testing::DoAll(::testing::SetArgPointee<4>(outputs),
                                   ::testing::Return(Status::OK())));
  } else {
    EXPECT_CALL(*mock, Run(_, _, _, _))
        .WillOnce(::testing::DoAll(::testing::SetArgPointee<3>(outputs),
                                   ::testing::Return(Status::OK())));
  }
  TF_ASSERT_OK(Create());
  *request_.mutable_input()->mutable_example_list()->mutable_examples()->Add() =
      example_with_output(2.0);
  const Status status = regressor_->Regress(request_, &result_);
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), ::testing::HasSubstr("output batch size"));
}

TEST_P(RegressorTest, UnexpectedOutputTensorType) {
  MockSession* mock = new MockSession;
  bundle_->session.reset(mock);
  // We expect a FLOAT output type; test returning a STRING.
  std::vector<Tensor> outputs = {Tensor(DT_STRING, TensorShape({1}))};
  if (GetParam()) {
    EXPECT_CALL(*mock, Run(_, _, _, _, _, _))
        .WillOnce(::testing::DoAll(::testing::SetArgPointee<4>(outputs),
                                   ::testing::Return(Status::OK())));
  } else {
    EXPECT_CALL(*mock, Run(_, _, _, _))
        .WillOnce(::testing::DoAll(::testing::SetArgPointee<3>(outputs),
                                   ::testing::Return(Status::OK())));
  }
  TF_ASSERT_OK(Create());
  *request_.mutable_input()->mutable_example_list()->mutable_examples()->Add() =
      example_with_output(2.0);
  const Status status = regressor_->Regress(request_, &result_);
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(),
              ::testing::HasSubstr("Expected output Tensor of DT_FLOAT"));
}

TEST_P(RegressorTest, MissingRegressionSignature) {
  tensorflow::serving::Signatures signatures;
  signatures.mutable_default_signature();
  TF_ASSERT_OK(tensorflow::serving::SetSignatures(signatures, meta_graph_def_));
  TF_ASSERT_OK(Create());
  Feature feature;
  feature.mutable_bytes_list()->add_value("uno");
  Example example;
  (*example.mutable_features()->mutable_feature())["class"] = feature;
  *request_.mutable_input()->mutable_example_list()->mutable_examples()->Add() =
      example;
  // TODO(b/26220896): This error should move to construction time.
  const Status status = regressor_->Regress(request_, &result_);
  ASSERT_FALSE(status.ok());
  // Old SessionBundle code treats a missing signature as a FAILED_PRECONDITION
  // but new SavedModel code treats it as an INVALID_ARGUMENT (signature
  // specified in the request was invalid).
  if (GetParam()) {
    EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, status.code()) << status;
  } else {
    EXPECT_EQ(::tensorflow::error::FAILED_PRECONDITION, status.code())
        << status;
  }
}

// Test all RegressorTest test cases with both SessionBundle and SavedModel.
INSTANTIATE_TEST_CASE_P(UseSavedModel, RegressorTest, ::testing::Bool());

}  // namespace
}  // namespace serving
}  // namespace tensorflow
