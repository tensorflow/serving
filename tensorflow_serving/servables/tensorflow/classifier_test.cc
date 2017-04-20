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

#include "tensorflow_serving/servables/tensorflow/classifier.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/map.h"
#include "tensorflow/contrib/session_bundle/bundle_shim.h"
#include "tensorflow/contrib/session_bundle/manifest.pb.h"
#include "tensorflow/contrib/session_bundle/session_bundle.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow_serving/apis/classification.pb.h"
#include "tensorflow_serving/apis/input.pb.h"
#include "tensorflow_serving/apis/model.pb.h"
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
const char kClassTensor[] = "output:0";
const char kOutputPlusOneClassTensor[] = "outputPlusOne:0";
const char kClassFeature[] = "class";
const char kScoreTensor[] = "score:0";
const char kScoreFeature[] = "score";
const char kImproperlySizedScoresTensor[] = "ImproperlySizedScores:0";

const char kOutputPlusOneSignature[] = "output_plus_one";
const char kInvalidNamedSignature[] = "invalid_regression_signature";
const char kImproperlySizedScoresSignature[] = "ImproperlySizedScoresSignature";

// Fake Session used for testing TensorFlowClassifier.
// Assumes the input Tensor "input:0" has serialized tensorflow::Example values.
// Copies the "class" bytes feature from each Example to be the classification
// class for that example.
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
    Tensor classes;
    Tensor scores;
    TF_RETURN_IF_ERROR(
        GetClassTensor(examples, output_names, &classes, &scores));
    for (const auto& output_name : output_names) {
      if (output_name == kClassTensor) {
        outputs->push_back(classes);
      } else if (output_name == kScoreTensor ||
                 output_name == kOutputPlusOneClassTensor) {
        outputs->push_back(scores);
      } else if (output_name == kImproperlySizedScoresTensor) {
        // Insert a rank 3 tensor which should be an error because scores are
        // expected to be rank 2.
        outputs->emplace_back(DT_FLOAT, TensorShape({scores.dim_size(0),
                                                     scores.dim_size(1), 10}));
      }
    }

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

  // Returns the number of individual elements in a Feature.
  static int FeatureSize(const Feature& feature) {
    if (feature.has_float_list()) {
      return feature.float_list().value_size();
    } else if (feature.has_int64_list()) {
      return feature.int64_list().value_size();
    } else if (feature.has_bytes_list()) {
      return feature.bytes_list().value_size();
    }
    return 0;
  }

  // Creates a Tensor by copying the "class" feature from each Example.
  // Requires each Example have an bytes feature called "class" which is of the
  // same non-zero length.
  static Status GetClassTensor(const std::vector<Example>& examples,
                               const std::vector<string>& output_names,
                               Tensor* classes, Tensor* scores) {
    if (examples.empty()) {
      return errors::Internal("empty example list");
    }

    auto iter = std::find(output_names.begin(), output_names.end(),
                          kOutputPlusOneClassTensor);
    const float offset = iter == output_names.end() ? 0 : 1;

    const int batch_size = examples.size();
    const int num_classes = FeatureSize(GetFeature(examples[0], kClassFeature));
    *classes = Tensor(DT_STRING, TensorShape({batch_size, num_classes}));
    *scores = Tensor(DT_FLOAT, TensorShape({batch_size, num_classes}));
    auto classes_matrix = classes->matrix<string>();
    auto scores_matrix = scores->matrix<float>();

    for (int i = 0; i < batch_size; ++i) {
      const Feature classes_feature = GetFeature(examples[i], kClassFeature);
      if (FeatureSize(classes_feature) != num_classes) {
        return errors::Internal("incorrect number of classes in feature: ",
                                classes_feature.DebugString());
      }
      const Feature scores_feature = GetFeature(examples[i], kScoreFeature);
      if (FeatureSize(scores_feature) != num_classes) {
        return errors::Internal("incorrect number of scores in feature: ",
                                scores_feature.DebugString());
      }
      for (int c = 0; c < num_classes; ++c) {
        classes_matrix(i, c) = classes_feature.bytes_list().value(c);
        scores_matrix(i, c) = scores_feature.float_list().value(c) + offset;
      }
    }
    return Status::OK();
  }

 private:
  const optional<int64> expected_timeout_;
};

// Add a named signature to the mutable signatures* parameter.
// If is_classification is false, will add a regression signature, which is
// invalid in classification requests.
void AddNamedSignature(const string& input_tensor_name,
                       const string& output_scores_tensor_name,
                       const string& signature_name,
                       const bool is_classification, Signatures* signatures) {
  tensorflow::serving::Signature named_signature;
  if (is_classification) {
    named_signature.mutable_classification_signature()
        ->mutable_input()
        ->set_tensor_name(input_tensor_name);
    named_signature.mutable_classification_signature()
        ->mutable_classes()
        ->set_tensor_name(kClassTensor);
    named_signature.mutable_classification_signature()
        ->mutable_scores()
        ->set_tensor_name(output_scores_tensor_name);
  } else {
    named_signature.mutable_regression_signature()
        ->mutable_input()
        ->set_tensor_name(input_tensor_name);
    named_signature.mutable_regression_signature()
        ->mutable_output()
        ->set_tensor_name(output_scores_tensor_name);
  }
  signatures->mutable_named_signatures()->insert(
      protobuf::MapPair<string, tensorflow::serving::Signature>(
          signature_name, named_signature));
}

// Parameter is 'bool use_saved_model'.
class ClassifierTest : public ::testing::TestWithParam<bool> {
 public:
  void SetUp() override {
    bundle_.reset(new SessionBundle);
    meta_graph_def_ = &bundle_->meta_graph_def;
    optional<int64> expected_timeout = GetRunOptions().timeout_in_ms();
    if (!UseSavedModel()) {
      // For SessionBundle we don't propagate the timeout.
      expected_timeout = nullopt;
    }
    fake_session_ = new FakeSession(expected_timeout);
    bundle_->session.reset(fake_session_);

    // Setup some defaults for our signature.
    tensorflow::serving::Signatures signatures;
    auto signature = signatures.mutable_default_signature()
                         ->mutable_classification_signature();
    signature->mutable_input()->set_tensor_name(kInputTensor);
    signature->mutable_classes()->set_tensor_name(kClassTensor);
    signature->mutable_scores()->set_tensor_name(kScoreTensor);

    AddNamedSignature(kInputTensor, kOutputPlusOneClassTensor,
                      kOutputPlusOneSignature, true /* is_classification */,
                      &signatures);
    AddNamedSignature(kInputTensor, kOutputPlusOneClassTensor,
                      kInvalidNamedSignature, false /* is_classification */,
                      &signatures);

    // Add a named signature where the output is not valid.
    AddNamedSignature(kInputTensor, kImproperlySizedScoresTensor,
                      kImproperlySizedScoresSignature,
                      true /* is_classification */, &signatures);
    TF_ASSERT_OK(
        tensorflow::serving::SetSignatures(signatures, meta_graph_def_));
  }

 protected:
  // Return an example with the feature "output" = [output].
  Example example(const std::vector<std::pair<string, float>>& class_scores) {
    Feature classes_feature;
    Feature scores_feature;
    for (const auto& class_score : class_scores) {
      classes_feature.mutable_bytes_list()->add_value(class_score.first);
      scores_feature.mutable_float_list()->add_value(class_score.second);
    }
    Example example;
    auto* features = example.mutable_features()->mutable_feature();
    (*features)[kClassFeature] = classes_feature;
    (*features)[kScoreFeature] = scores_feature;
    return example;
  }

  // Whether or not to use SavedModel for this test. Simply wraps GetParam()
  // with a more meaningful name.
  bool UseSavedModel() { return GetParam(); }

  Status Create() {
    if (UseSavedModel()) {
      std::unique_ptr<SavedModelBundle> saved_model(new SavedModelBundle);
      TF_CHECK_OK(internal::ConvertSessionBundleToSavedModelBundle(
          *bundle_, saved_model.get()));
      return CreateClassifierFromSavedModelBundle(
          GetRunOptions(), std::move(saved_model), &classifier_);
    } else {
      return CreateClassifierFromBundle(std::move(bundle_), &classifier_);
    }
  }

  // Variables used to create the classifier.
  tensorflow::MetaGraphDef* meta_graph_def_;
  FakeSession* fake_session_;
  std::unique_ptr<SessionBundle> bundle_;

  // Classifier valid after calling create.
  std::unique_ptr<ClassifierInterface> classifier_;

  // Convenience variables.
  ClassificationRequest request_;
  ClassificationResult result_;

 private:
  RunOptions GetRunOptions() const {
    RunOptions run_options;
    run_options.set_timeout_in_ms(42);
    return run_options;
  }
};

TEST_P(ClassifierTest, ExampleList) {
  TF_ASSERT_OK(Create());
  auto* examples =
      request_.mutable_input()->mutable_example_list()->mutable_examples();
  *examples->Add() = example({{"dos", 2}, {"uno", 1}});
  *examples->Add() = example({{"cuatro", 4}, {"tres", 3}});
  TF_ASSERT_OK(classifier_->Classify(request_, &result_));
  EXPECT_THAT(result_, EqualsProto(" classifications { "
                                   "   classes { "
                                   "     label: 'dos' "
                                   "     score: 2 "
                                   "   } "
                                   "   classes { "
                                   "     label: 'uno' "
                                   "     score: 1 "
                                   "   } "
                                   " } "
                                   " classifications { "
                                   "   classes { "
                                   "     label: 'cuatro' "
                                   "     score: 4 "
                                   "   } "
                                   "   classes { "
                                   "     label: 'tres' "
                                   "     score: 3 "
                                   "   } "
                                   " } "));
}

TEST_P(ClassifierTest, ExampleListWithContext) {
  TF_ASSERT_OK(Create());
  auto* list_and_context =
      request_.mutable_input()->mutable_example_list_with_context();
  // Context gets copied to each example.
  *list_and_context->mutable_context() = example({{"dos", 2}, {"uno", 1}});
  // Add empty examples to recieve the context.
  list_and_context->add_examples();
  list_and_context->add_examples();
  TF_ASSERT_OK(classifier_->Classify(request_, &result_));
  EXPECT_THAT(result_, EqualsProto(" classifications { "
                                   "   classes { "
                                   "     label: 'dos' "
                                   "     score: 2 "
                                   "   } "
                                   "   classes { "
                                   "     label: 'uno' "
                                   "     score: 1 "
                                   "   } "
                                   " } "
                                   " classifications { "
                                   "   classes { "
                                   "     label: 'dos' "
                                   "     score: 2 "
                                   "   } "
                                   "   classes { "
                                   "     label: 'uno' "
                                   "     score: 1 "
                                   "   } "
                                   " } "));
}

TEST_P(ClassifierTest, ExampleListWithContext_DuplicateFeatures) {
  TF_ASSERT_OK(Create());
  auto* list_and_context =
      request_.mutable_input()->mutable_example_list_with_context();
  // Context gets copied to each example.
  *list_and_context->mutable_context() = example({{"uno", 1}, {"dos", 2}});
  // Add an empty example, after merge it should be equal to the context.
  list_and_context->add_examples();
  // Add an example with a duplicate feature.  Technically this behavior is
  // undefined so here we are ensuring we don't crash.
  *list_and_context->add_examples() = example({{"tres", 3}, {"cuatro", 4}});
  TF_ASSERT_OK(classifier_->Classify(request_, &result_));
  EXPECT_THAT(result_, EqualsProto(" classifications { "
                                   "   classes { "
                                   "     label: 'uno' "
                                   "     score: 1 "
                                   "   } "
                                   "   classes { "
                                   "     label: 'dos' "
                                   "     score: 2 "
                                   "   } "
                                   " } "
                                   " classifications { "
                                   "   classes { "
                                   "     label: 'tres' "
                                   "     score: 3 "
                                   "   } "
                                   "   classes { "
                                   "     label: 'cuatro' "
                                   "     score: 4 "
                                   "   } "
                                   " } "));
}

TEST_P(ClassifierTest, ClassesOnly) {
  tensorflow::serving::Signatures signatures;
  auto signature = signatures.mutable_default_signature()
                       ->mutable_classification_signature();
  signature->mutable_input()->set_tensor_name(kInputTensor);
  signature->mutable_classes()->set_tensor_name(kClassTensor);
  // No scores Tensor.
  TF_ASSERT_OK(tensorflow::serving::SetSignatures(signatures, meta_graph_def_));
  TF_ASSERT_OK(Create());
  auto* examples =
      request_.mutable_input()->mutable_example_list()->mutable_examples();
  *examples->Add() = example({{"dos", 2}, {"uno", 1}});
  *examples->Add() = example({{"cuatro", 4}, {"tres", 3}});
  TF_ASSERT_OK(classifier_->Classify(request_, &result_));
  EXPECT_THAT(result_, EqualsProto(" classifications { "
                                   "   classes { "
                                   "     label: 'dos' "
                                   "   } "
                                   "   classes { "
                                   "     label: 'uno' "
                                   "   } "
                                   " } "
                                   " classifications { "
                                   "   classes { "
                                   "     label: 'cuatro' "
                                   "   } "
                                   "   classes { "
                                   "     label: 'tres' "
                                   "   } "
                                   " } "));
}

TEST_P(ClassifierTest, ScoresOnly) {
  tensorflow::serving::Signatures signatures;
  auto signature = signatures.mutable_default_signature()
                       ->mutable_classification_signature();
  signature->mutable_input()->set_tensor_name(kInputTensor);
  // No classes Tensor.
  signature->mutable_scores()->set_tensor_name(kScoreTensor);
  TF_ASSERT_OK(tensorflow::serving::SetSignatures(signatures, meta_graph_def_));
  TF_ASSERT_OK(Create());
  auto* examples =
      request_.mutable_input()->mutable_example_list()->mutable_examples();
  *examples->Add() = example({{"dos", 2}, {"uno", 1}});
  *examples->Add() = example({{"cuatro", 4}, {"tres", 3}});
  TF_ASSERT_OK(classifier_->Classify(request_, &result_));
  EXPECT_THAT(result_, EqualsProto(" classifications { "
                                   "   classes { "
                                   "     score: 2 "
                                   "   } "
                                   "   classes { "
                                   "     score: 1 "
                                   "   } "
                                   " } "
                                   " classifications { "
                                   "   classes { "
                                   "     score: 4 "
                                   "   } "
                                   "   classes { "
                                   "     score: 3 "
                                   "   } "
                                   " } "));
}

TEST_P(ClassifierTest, ValidNamedSignature) {
  TF_ASSERT_OK(Create());
  request_.mutable_model_spec()->set_signature_name(kOutputPlusOneSignature);
  auto* examples =
      request_.mutable_input()->mutable_example_list()->mutable_examples();
  *examples->Add() = example({{"dos", 2}, {"uno", 1}});
  *examples->Add() = example({{"cuatro", 4}, {"tres", 3}});
  TF_ASSERT_OK(classifier_->Classify(request_, &result_));

  // If using saved_model, this test should use the kOutputPlusOneSignature
  // named signature. Otherwise, when using session_bundle, the signature_name
  // in the model_spec will be ignored and the default signature will be used.
  if (UseSavedModel()) {
    EXPECT_THAT(result_, EqualsProto(" classifications { "
                                     "   classes { "
                                     "     label: 'dos' "
                                     "     score: 3 "
                                     "   } "
                                     "   classes { "
                                     "     label: 'uno' "
                                     "     score: 2 "
                                     "   } "
                                     " } "
                                     " classifications { "
                                     "   classes { "
                                     "     label: 'cuatro' "
                                     "     score: 5 "
                                     "   } "
                                     "   classes { "
                                     "     label: 'tres' "
                                     "     score: 4 "
                                     "   } "
                                     " } "));
  } else {
    EXPECT_THAT(result_, EqualsProto(" classifications { "
                                     "   classes { "
                                     "     label: 'dos' "
                                     "     score: 2 "
                                     "   } "
                                     "   classes { "
                                     "     label: 'uno' "
                                     "     score: 1 "
                                     "   } "
                                     " } "
                                     " classifications { "
                                     "   classes { "
                                     "     label: 'cuatro' "
                                     "     score: 4 "
                                     "   } "
                                     "   classes { "
                                     "     label: 'tres' "
                                     "     score: 3 "
                                     "   } "
                                     " } "));
  }
}

TEST_P(ClassifierTest, InvalidNamedSignature) {
  TF_ASSERT_OK(Create());
  request_.mutable_model_spec()->set_signature_name(kInvalidNamedSignature);
  auto* examples =
      request_.mutable_input()->mutable_example_list()->mutable_examples();
  *examples->Add() = example({{"dos", 2}, {"uno", 1}});
  *examples->Add() = example({{"cuatro", 4}, {"tres", 3}});
  const Status status = classifier_->Classify(request_, &result_);

  // If using saved_model, this test should fail because the named_signature
  // requested is actually a regression signature. When using session_bundle,
  // the signature_name will be ignored and the default signature will be used.
  if (UseSavedModel()) {
    ASSERT_FALSE(status.ok());
    EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, status.code()) << status;
  } else {
    TF_ASSERT_OK(status);
    EXPECT_THAT(result_, EqualsProto(" classifications { "
                                     "   classes { "
                                     "     label: 'dos' "
                                     "     score: 2 "
                                     "   } "
                                     "   classes { "
                                     "     label: 'uno' "
                                     "     score: 1 "
                                     "   } "
                                     " } "
                                     " classifications { "
                                     "   classes { "
                                     "     label: 'cuatro' "
                                     "     score: 4 "
                                     "   } "
                                     "   classes { "
                                     "     label: 'tres' "
                                     "     score: 3 "
                                     "   } "
                                     " } "));
  }
}

TEST_P(ClassifierTest, MalformedScores) {
  // If not using SavedModel, we don't use named signatures so the test is not
  // actually testing the right thing. Skip it.
  if (!UseSavedModel()) {
    return;
  }

  TF_ASSERT_OK(Create());
  request_.mutable_model_spec()->set_signature_name(
      kImproperlySizedScoresSignature);
  auto* examples =
      request_.mutable_input()->mutable_example_list()->mutable_examples();
  *examples->Add() = example({{"dos", 2}, {"uno", 1}});
  *examples->Add() = example({{"cuatro", 4}, {"tres", 3}});
  const Status status = classifier_->Classify(request_, &result_);

  ASSERT_FALSE(status.ok());
  EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, status.code()) << status;
}

TEST_P(ClassifierTest, MissingClassificationSignature) {
  tensorflow::serving::Signatures signatures;
  signatures.mutable_default_signature();
  TF_ASSERT_OK(tensorflow::serving::SetSignatures(signatures, meta_graph_def_));
  TF_ASSERT_OK(Create());
  auto* examples =
      request_.mutable_input()->mutable_example_list()->mutable_examples();
  *examples->Add() = example({{"dos", 2}});
  // TODO(b/26220896): This error should move to construction time.
  const Status status = classifier_->Classify(request_, &result_);
  ASSERT_FALSE(status.ok());
  // Old SessionBundle code treats a missing signature as a FAILED_PRECONDITION
  // but new SavedModel code treats it as an INVALID_ARGUMENT (signature
  // specified in the request was invalid).
  if (UseSavedModel()) {
    EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, status.code()) << status;
  } else {
    EXPECT_EQ(::tensorflow::error::FAILED_PRECONDITION, status.code())
        << status;
  }
}

TEST_P(ClassifierTest, EmptyInput) {
  TF_ASSERT_OK(Create());
  // Touch input.
  request_.mutable_input();
  const Status status = classifier_->Classify(request_, &result_);
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(),
              ::testing::HasSubstr("Invalid argument: Input is empty"));
}

TEST_P(ClassifierTest, EmptyExampleList) {
  TF_ASSERT_OK(Create());
  // Touch ExampleList.
  request_.mutable_input()->mutable_example_list();
  const Status status = classifier_->Classify(request_, &result_);
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(),
              ::testing::HasSubstr("Invalid argument: Input is empty"));
}

TEST_P(ClassifierTest, EmptyExampleListWithContext) {
  TF_ASSERT_OK(Create());
  // Touch ExampleListWithContext, context populated but no Examples.
  *request_.mutable_input()
       ->mutable_example_list_with_context()
       ->mutable_context() = example({{"dos", 2}});
  const Status status = classifier_->Classify(request_, &result_);
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(),
              ::testing::HasSubstr("Invalid argument: Input is empty"));
}

TEST_P(ClassifierTest, RunsFails) {
  MockSession* mock = new MockSession;
  bundle_->session.reset(mock);
  if (UseSavedModel()) {
    EXPECT_CALL(*mock, Run(_, _, _, _, _, _))
        .WillRepeatedly(
            ::testing::Return(errors::Internal("Run totally failed")));
  } else {
    EXPECT_CALL(*mock, Run(_, _, _, _))
        .WillRepeatedly(
            ::testing::Return(errors::Internal("Run totally failed")));
  }
  TF_ASSERT_OK(Create());
  auto* examples =
      request_.mutable_input()->mutable_example_list()->mutable_examples();
  *examples->Add() = example({{"dos", 2}});
  const Status status = classifier_->Classify(request_, &result_);
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), ::testing::HasSubstr("Run totally failed"));
}

TEST_P(ClassifierTest, ClassesIncorrectTensorBatchSize) {
  MockSession* mock = new MockSession;
  bundle_->session.reset(mock);
  // This Tensor only has one batch item but we will have two inputs.
  Tensor classes(DT_STRING, TensorShape({1, 2}));
  Tensor scores(DT_FLOAT, TensorShape({2, 2}));
  std::vector<Tensor> outputs = {classes, scores};
  if (UseSavedModel()) {
    EXPECT_CALL(*mock, Run(_, _, _, _, _, _))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgPointee<4>(outputs),
                                         ::testing::Return(Status::OK())));
  } else {
    EXPECT_CALL(*mock, Run(_, _, _, _))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgPointee<3>(outputs),
                                         ::testing::Return(Status::OK())));
  }
  TF_ASSERT_OK(Create());
  auto* examples =
      request_.mutable_input()->mutable_example_list()->mutable_examples();
  *examples->Add() = example({{"dos", 2}, {"uno", 1}});
  *examples->Add() = example({{"cuatro", 4}, {"tres", 3}});

  const Status status = classifier_->Classify(request_, &result_);
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), ::testing::HasSubstr("batch size"));
}

TEST_P(ClassifierTest, ClassesIncorrectTensorType) {
  MockSession* mock = new MockSession;
  bundle_->session.reset(mock);
  // This Tensor is the wrong type for class.
  Tensor classes(DT_FLOAT, TensorShape({2, 2}));
  Tensor scores(DT_FLOAT, TensorShape({2, 2}));
  std::vector<Tensor> outputs = {classes, scores};
  if (UseSavedModel()) {
    EXPECT_CALL(*mock, Run(_, _, _, _, _, _))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgPointee<4>(outputs),
                                         ::testing::Return(Status::OK())));
  } else {
    EXPECT_CALL(*mock, Run(_, _, _, _))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgPointee<3>(outputs),
                                         ::testing::Return(Status::OK())));
  }
  TF_ASSERT_OK(Create());
  auto* examples =
      request_.mutable_input()->mutable_example_list()->mutable_examples();
  *examples->Add() = example({{"dos", 2}, {"uno", 1}});
  *examples->Add() = example({{"cuatro", 4}, {"tres", 3}});

  const Status status = classifier_->Classify(request_, &result_);
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(),
              ::testing::HasSubstr("Expected classes Tensor of DT_STRING"));
}

TEST_P(ClassifierTest, ScoresIncorrectTensorBatchSize) {
  MockSession* mock = new MockSession;
  bundle_->session.reset(mock);
  Tensor classes(DT_STRING, TensorShape({2, 2}));
  // This Tensor only has one batch item but we will have two inputs.
  Tensor scores(DT_FLOAT, TensorShape({1, 2}));
  std::vector<Tensor> outputs = {classes, scores};
  if (UseSavedModel()) {
    EXPECT_CALL(*mock, Run(_, _, _, _, _, _))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgPointee<4>(outputs),
                                         ::testing::Return(Status::OK())));
  } else {
    EXPECT_CALL(*mock, Run(_, _, _, _))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgPointee<3>(outputs),
                                         ::testing::Return(Status::OK())));
  }
  TF_ASSERT_OK(Create());
  auto* examples =
      request_.mutable_input()->mutable_example_list()->mutable_examples();
  *examples->Add() = example({{"dos", 2}, {"uno", 1}});
  *examples->Add() = example({{"cuatro", 4}, {"tres", 3}});

  const Status status = classifier_->Classify(request_, &result_);
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), ::testing::HasSubstr("batch size"));
}

TEST_P(ClassifierTest, ScoresIncorrectTensorType) {
  MockSession* mock = new MockSession;
  bundle_->session.reset(mock);
  Tensor classes(DT_STRING, TensorShape({2, 2}));
  // This Tensor is the wrong type for class.
  Tensor scores(DT_STRING, TensorShape({2, 2}));
  std::vector<Tensor> outputs = {classes, scores};
  if (UseSavedModel()) {
    EXPECT_CALL(*mock, Run(_, _, _, _, _, _))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgPointee<4>(outputs),
                                         ::testing::Return(Status::OK())));
  } else {
    EXPECT_CALL(*mock, Run(_, _, _, _))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgPointee<3>(outputs),
                                         ::testing::Return(Status::OK())));
  }
  TF_ASSERT_OK(Create());
  auto* examples =
      request_.mutable_input()->mutable_example_list()->mutable_examples();
  *examples->Add() = example({{"dos", 2}, {"uno", 1}});
  *examples->Add() = example({{"cuatro", 4}, {"tres", 3}});

  const Status status = classifier_->Classify(request_, &result_);
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(),
              ::testing::HasSubstr("Expected scores Tensor of DT_FLOAT"));
}

TEST_P(ClassifierTest, MismatchedNumberOfTensorClasses) {
  MockSession* mock = new MockSession;
  bundle_->session.reset(mock);
  Tensor classes(DT_STRING, TensorShape({2, 2}));
  // Scores Tensor has three scores but classes only has two labels.
  Tensor scores(DT_FLOAT, TensorShape({2, 3}));
  std::vector<Tensor> outputs = {classes, scores};
  if (UseSavedModel()) {
    EXPECT_CALL(*mock, Run(_, _, _, _, _, _))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgPointee<4>(outputs),
                                         ::testing::Return(Status::OK())));
  } else {
    EXPECT_CALL(*mock, Run(_, _, _, _))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgPointee<3>(outputs),
                                         ::testing::Return(Status::OK())));
  }
  TF_ASSERT_OK(Create());
  auto* examples =
      request_.mutable_input()->mutable_example_list()->mutable_examples();
  *examples->Add() = example({{"dos", 2}, {"uno", 1}});
  *examples->Add() = example({{"cuatro", 4}, {"tres", 3}});

  const Status status = classifier_->Classify(request_, &result_);
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(
      status.ToString(),
      ::testing::HasSubstr(
          "Tensors class and score should match in dim_size(1). Got 2 vs. 3"));
}

// Test all ClassifierTest test cases with both SessionBundle and SavedModel.
INSTANTIATE_TEST_CASE_P(UseSavedModel, ClassifierTest, ::testing::Bool());

}  // namespace
}  // namespace serving
}  // namespace tensorflow
