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
#include "tensorflow_serving/apis/classification.pb.h"
#include "tensorflow_serving/apis/input.pb.h"
#include "tensorflow_serving/apis/model.pb.h"
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
    auto classes_matrix = classes->matrix<tstring>();
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
    return OkStatus();
  }

 private:
  const absl::optional<int64_t> expected_timeout_;
};

class ClassifierTest : public ::testing::TestWithParam<bool> {
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
    (*sig_def.mutable_inputs())[kClassifyInputs] = input_tensor_info;
    TensorInfo class_tensor_info;
    class_tensor_info.set_name(kClassTensor);
    (*sig_def.mutable_outputs())[kClassifyOutputClasses] = class_tensor_info;
    TensorInfo scores_tensor_info;
    scores_tensor_info.set_name(kScoreTensor);
    (*sig_def.mutable_outputs())[kClassifyOutputScores] = scores_tensor_info;
    if (IsMethodNameCheckEnabled())
      sig_def.set_method_name(kClassifyMethodName);
    (*signature_defs)[kDefaultServingSignatureDefKey] = sig_def;

    AddNamedSignatureToSavedModelBundle(
        kInputTensor, kOutputPlusOneClassTensor, kOutputPlusOneSignature,
        true /* is_classification */, meta_graph_def_);
    AddNamedSignatureToSavedModelBundle(
        kInputTensor, kOutputPlusOneClassTensor, kInvalidNamedSignature,
        false /* is_classification */, meta_graph_def_);

    // Add a named signature where the output is not valid.
    AddNamedSignatureToSavedModelBundle(
        kInputTensor, kImproperlySizedScoresTensor,
        kImproperlySizedScoresSignature, true /* is_classification */,
        meta_graph_def_);
  }

 protected:
  bool IsMethodNameCheckEnabled() { return GetParam(); }

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

  Status Create() {
    std::unique_ptr<SavedModelBundle> saved_model(new SavedModelBundle);
    saved_model->meta_graph_def = saved_model_bundle_->meta_graph_def;
    saved_model->session = std::move(saved_model_bundle_->session);
    return CreateClassifierFromSavedModelBundle(
        GetRunOptions(), std::move(saved_model), &classifier_);
  }

  RunOptions GetRunOptions() const {
    RunOptions run_options;
    run_options.set_timeout_in_ms(42);
    return run_options;
  }

  // Add a named signature to the mutable meta_graph_def* parameter.
  // If is_classification is false, will add a regression signature, which is
  // invalid in classification requests.
  void AddNamedSignatureToSavedModelBundle(
      const string& input_tensor_name, const string& output_scores_tensor_name,
      const string& signature_name, const bool is_classification,
      tensorflow::MetaGraphDef* meta_graph_def) {
    auto* signature_defs = meta_graph_def->mutable_signature_def();
    SignatureDef sig_def;
    TensorInfo input_tensor_info;
    input_tensor_info.set_name(input_tensor_name);
    string method_name;
    (*sig_def.mutable_inputs())[kClassifyInputs] = input_tensor_info;
    if (is_classification) {
      TensorInfo scores_tensor_info;
      scores_tensor_info.set_name(output_scores_tensor_name);
      (*sig_def.mutable_outputs())[kClassifyOutputScores] = scores_tensor_info;
      TensorInfo class_tensor_info;
      class_tensor_info.set_name(kClassTensor);
      (*sig_def.mutable_outputs())[kClassifyOutputClasses] = class_tensor_info;
      method_name = kClassifyMethodName;
    } else {
      TensorInfo output_tensor_info;
      output_tensor_info.set_name(output_scores_tensor_name);
      (*sig_def.mutable_outputs())[kRegressOutputs] = output_tensor_info;
      method_name = kRegressMethodName;
    }
    if (IsMethodNameCheckEnabled()) sig_def.set_method_name(method_name);
    (*signature_defs)[signature_name] = sig_def;
  }

  // Variables used to create the classifier.
  tensorflow::MetaGraphDef* meta_graph_def_;
  FakeSession* fake_session_;
  std::unique_ptr<SavedModelBundle> saved_model_bundle_;

  // Classifier valid after calling create.
  std::unique_ptr<ClassifierInterface> classifier_;

  // Convenience variables.
  ClassificationRequest request_;
  ClassificationResult result_;
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
  // Test RunClassify
  ClassificationResponse response;
  TF_ASSERT_OK(RunClassify(GetRunOptions(), saved_model_bundle_->meta_graph_def,
                           {}, fake_session_, request_, &response));
  EXPECT_THAT(response.result(), EqualsProto(" classifications { "
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

  ClassificationResponse response;
  TF_ASSERT_OK(RunClassify(GetRunOptions(), saved_model_bundle_->meta_graph_def,
                           {}, fake_session_, request_, &response));
  EXPECT_THAT(response.result(), EqualsProto(" classifications { "
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

  ClassificationResponse response;
  TF_ASSERT_OK(RunClassify(GetRunOptions(), saved_model_bundle_->meta_graph_def,
                           {}, fake_session_, request_, &response));
  EXPECT_THAT(response.result(), EqualsProto(" classifications { "
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
  auto* signature_defs = meta_graph_def_->mutable_signature_def();
  (*signature_defs)[kDefaultServingSignatureDefKey].mutable_outputs()->erase(
      kClassifyOutputScores);
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

  ClassificationResponse response;
  TF_ASSERT_OK(RunClassify(GetRunOptions(), saved_model_bundle_->meta_graph_def,
                           {}, fake_session_, request_, &response));
  EXPECT_THAT(response.result(), EqualsProto(" classifications { "
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
  auto* signature_defs = meta_graph_def_->mutable_signature_def();
  (*signature_defs)[kDefaultServingSignatureDefKey].mutable_outputs()->erase(
      kClassifyOutputClasses);

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

  ClassificationResponse response;
  TF_ASSERT_OK(RunClassify(GetRunOptions(), saved_model_bundle_->meta_graph_def,
                           {}, fake_session_, request_, &response));
  EXPECT_THAT(response.result(), EqualsProto(" classifications { "
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

TEST_P(ClassifierTest, ZeroScoresArePresent) {
  auto* signature_defs = meta_graph_def_->mutable_signature_def();
  (*signature_defs)[kDefaultServingSignatureDefKey].mutable_outputs()->erase(
      kClassifyOutputClasses);
  TF_ASSERT_OK(Create());
  auto* examples =
      request_.mutable_input()->mutable_example_list()->mutable_examples();
  *examples->Add() = example({{"minus", -1}, {"zero", 0}, {"one", 1}});
  const std::vector<double> expected_outputs = {-1, 0, 1};

  TF_ASSERT_OK(classifier_->Classify(request_, &result_));
  // Parse the protos and compare the results with expected scores.
  ASSERT_EQ(result_.classifications_size(), 1);
  auto& classification = result_.classifications(0);
  ASSERT_EQ(classification.classes_size(), 3);

  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(classification.classes(i).score(), expected_outputs[i], 1e-7);
  }

  ClassificationResponse response;
  TF_ASSERT_OK(RunClassify(GetRunOptions(), saved_model_bundle_->meta_graph_def,
                           {}, fake_session_, request_, &response));
  // Parse the protos and compare the results with expected scores.
  ASSERT_EQ(response.result().classifications_size(), 1);
  auto& classification_resp = result_.classifications(0);
  ASSERT_EQ(classification_resp.classes_size(), 3);

  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(classification_resp.classes(i).score(), expected_outputs[i],
                1e-7);
  }
}

TEST_P(ClassifierTest, ValidNamedSignature) {
  TF_ASSERT_OK(Create());
  request_.mutable_model_spec()->set_signature_name(kOutputPlusOneSignature);
  auto* examples =
      request_.mutable_input()->mutable_example_list()->mutable_examples();
  *examples->Add() = example({{"dos", 2}, {"uno", 1}});
  *examples->Add() = example({{"cuatro", 4}, {"tres", 3}});
  TF_ASSERT_OK(classifier_->Classify(request_, &result_));

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

  ClassificationResponse response;
  TF_ASSERT_OK(RunClassify(GetRunOptions(), saved_model_bundle_->meta_graph_def,
                           {}, fake_session_, request_, &response));
  EXPECT_THAT(response.result(), EqualsProto(" classifications { "
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
}

TEST_P(ClassifierTest, InvalidNamedSignature) {
  TF_ASSERT_OK(Create());
  request_.mutable_model_spec()->set_signature_name(kInvalidNamedSignature);
  auto* examples =
      request_.mutable_input()->mutable_example_list()->mutable_examples();
  *examples->Add() = example({{"dos", 2}, {"uno", 1}});
  *examples->Add() = example({{"cuatro", 4}, {"tres", 3}});
  Status status = classifier_->Classify(request_, &result_);

  ASSERT_FALSE(status.ok());
  EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, status.code()) << status;

  ClassificationResponse response;
  status = RunClassify(GetRunOptions(), saved_model_bundle_->meta_graph_def, {},
                       fake_session_, request_, &response);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, status.code()) << status;
}

TEST_P(ClassifierTest, MalformedScores) {
  TF_ASSERT_OK(Create());
  request_.mutable_model_spec()->set_signature_name(
      kImproperlySizedScoresSignature);
  auto* examples =
      request_.mutable_input()->mutable_example_list()->mutable_examples();
  *examples->Add() = example({{"dos", 2}, {"uno", 1}});
  *examples->Add() = example({{"cuatro", 4}, {"tres", 3}});
  Status status = classifier_->Classify(request_, &result_);

  ASSERT_FALSE(status.ok());
  EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, status.code()) << status;

  ClassificationResponse response;
  status = RunClassify(GetRunOptions(), saved_model_bundle_->meta_graph_def, {},
                       fake_session_, request_, &response);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, status.code()) << status;
}

TEST_P(ClassifierTest, MissingClassificationSignature) {
  auto* signature_defs = meta_graph_def_->mutable_signature_def();
  SignatureDef sig_def;
  (*signature_defs)[kDefaultServingSignatureDefKey] = sig_def;
  TF_ASSERT_OK(Create());
  auto* examples =
      request_.mutable_input()->mutable_example_list()->mutable_examples();
  *examples->Add() = example({{"dos", 2}});
  // TODO(b/26220896): This error should move to construction time.
  Status status = classifier_->Classify(request_, &result_);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, status.code()) << status;

  ClassificationResponse response;
  status = RunClassify(GetRunOptions(), saved_model_bundle_->meta_graph_def, {},
                       fake_session_, request_, &response);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, status.code()) << status;
}

TEST_P(ClassifierTest, EmptyInput) {
  TF_ASSERT_OK(Create());
  // Touch input.
  request_.mutable_input();
  Status status = classifier_->Classify(request_, &result_);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.code(), error::Code::INVALID_ARGUMENT);
  EXPECT_THAT(status.error_message(), ::testing::HasSubstr("Input is empty"));

  ClassificationResponse response;
  status = RunClassify(GetRunOptions(), saved_model_bundle_->meta_graph_def, {},
                       fake_session_, request_, &response);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.code(), error::Code::INVALID_ARGUMENT);
  EXPECT_THAT(status.error_message(), ::testing::HasSubstr("Input is empty"));
}

TEST_P(ClassifierTest, EmptyExampleList) {
  TF_ASSERT_OK(Create());
  // Touch ExampleList.
  request_.mutable_input()->mutable_example_list();
  Status status = classifier_->Classify(request_, &result_);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.code(), error::Code::INVALID_ARGUMENT);
  EXPECT_THAT(status.error_message(), ::testing::HasSubstr("Input is empty"));

  ClassificationResponse response;
  status = RunClassify(GetRunOptions(), saved_model_bundle_->meta_graph_def, {},
                       fake_session_, request_, &response);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.code(), error::Code::INVALID_ARGUMENT);
  EXPECT_THAT(status.error_message(), ::testing::HasSubstr("Input is empty"));
}

TEST_P(ClassifierTest, EmptyExampleListWithContext) {
  TF_ASSERT_OK(Create());
  // Touch ExampleListWithContext, context populated but no Examples.
  *request_.mutable_input()
       ->mutable_example_list_with_context()
       ->mutable_context() = example({{"dos", 2}});
  Status status = classifier_->Classify(request_, &result_);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.code(), error::Code::INVALID_ARGUMENT);
  EXPECT_THAT(status.error_message(), ::testing::HasSubstr("Input is empty"));

  ClassificationResponse response;
  status = RunClassify(GetRunOptions(), saved_model_bundle_->meta_graph_def, {},
                       fake_session_, request_, &response);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.code(), error::Code::INVALID_ARGUMENT);
  EXPECT_THAT(status.error_message(), ::testing::HasSubstr("Input is empty"));
}

TEST_P(ClassifierTest, RunsFails) {
  MockSession* mock = new MockSession;
  saved_model_bundle_->session.reset(mock);
  EXPECT_CALL(*mock, Run(_, _, _, _, _, _, _))
      .WillRepeatedly(
          ::testing::Return(errors::Internal("Run totally failed")));
  TF_ASSERT_OK(Create());
  auto* examples =
      request_.mutable_input()->mutable_example_list()->mutable_examples();
  *examples->Add() = example({{"dos", 2}});
  Status status = classifier_->Classify(request_, &result_);
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), ::testing::HasSubstr("Run totally failed"));

  ClassificationResponse response;
  status = RunClassify(GetRunOptions(), saved_model_bundle_->meta_graph_def, {},
                       mock, request_, &response);
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), ::testing::HasSubstr("Run totally failed"));
}

TEST_P(ClassifierTest, ClassesIncorrectTensorBatchSize) {
  MockSession* mock = new MockSession;
  saved_model_bundle_->session.reset(mock);
  // This Tensor only has one batch item but we will have two inputs.
  Tensor classes(DT_STRING, TensorShape({1, 2}));
  Tensor scores(DT_FLOAT, TensorShape({2, 2}));
  std::vector<Tensor> outputs = {classes, scores};
  EXPECT_CALL(*mock, Run(_, _, _, _, _, _, _))
      .WillRepeatedly(::testing::DoAll(::testing::SetArgPointee<4>(outputs),
                                       ::testing::Return(OkStatus())));
  TF_ASSERT_OK(Create());
  auto* examples =
      request_.mutable_input()->mutable_example_list()->mutable_examples();
  *examples->Add() = example({{"dos", 2}, {"uno", 1}});
  *examples->Add() = example({{"cuatro", 4}, {"tres", 3}});

  Status status = classifier_->Classify(request_, &result_);
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), ::testing::HasSubstr("batch size"));

  ClassificationResponse response;
  status = RunClassify(GetRunOptions(), saved_model_bundle_->meta_graph_def, {},
                       mock, request_, &response);
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), ::testing::HasSubstr("batch size"));
}

TEST_P(ClassifierTest, ClassesIncorrectTensorType) {
  MockSession* mock = new MockSession;
  saved_model_bundle_->session.reset(mock);

  // This Tensor is the wrong type for class.
  Tensor classes(DT_FLOAT, TensorShape({2, 2}));
  Tensor scores(DT_FLOAT, TensorShape({2, 2}));
  std::vector<Tensor> outputs = {classes, scores};
  EXPECT_CALL(*mock, Run(_, _, _, _, _, _, _))
      .WillRepeatedly(::testing::DoAll(::testing::SetArgPointee<4>(outputs),
                                       ::testing::Return(OkStatus())));
  TF_ASSERT_OK(Create());
  auto* examples =
      request_.mutable_input()->mutable_example_list()->mutable_examples();
  *examples->Add() = example({{"dos", 2}, {"uno", 1}});
  *examples->Add() = example({{"cuatro", 4}, {"tres", 3}});

  Status status = classifier_->Classify(request_, &result_);
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(),
              ::testing::HasSubstr("Expected classes Tensor of DT_STRING"));
  ClassificationResponse response;
  status = RunClassify(GetRunOptions(), saved_model_bundle_->meta_graph_def, {},
                       mock, request_, &response);
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(),
              ::testing::HasSubstr("Expected classes Tensor of DT_STRING"));
}

TEST_P(ClassifierTest, ScoresIncorrectTensorBatchSize) {
  MockSession* mock = new MockSession;
  saved_model_bundle_->session.reset(mock);
  Tensor classes(DT_STRING, TensorShape({2, 2}));
  // This Tensor only has one batch item but we will have two inputs.
  Tensor scores(DT_FLOAT, TensorShape({1, 2}));
  std::vector<Tensor> outputs = {classes, scores};
  EXPECT_CALL(*mock, Run(_, _, _, _, _, _, _))
      .WillRepeatedly(::testing::DoAll(::testing::SetArgPointee<4>(outputs),
                                       ::testing::Return(OkStatus())));
  TF_ASSERT_OK(Create());
  auto* examples =
      request_.mutable_input()->mutable_example_list()->mutable_examples();
  *examples->Add() = example({{"dos", 2}, {"uno", 1}});
  *examples->Add() = example({{"cuatro", 4}, {"tres", 3}});

  Status status = classifier_->Classify(request_, &result_);
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), ::testing::HasSubstr("batch size"));

  ClassificationResponse response;
  status = RunClassify(GetRunOptions(), saved_model_bundle_->meta_graph_def, {},
                       mock, request_, &response);
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), ::testing::HasSubstr("batch size"));
}

TEST_P(ClassifierTest, ScoresIncorrectTensorType) {
  MockSession* mock = new MockSession;
  saved_model_bundle_->session.reset(mock);
  Tensor classes(DT_STRING, TensorShape({2, 2}));
  // This Tensor is the wrong type for class.
  Tensor scores(DT_STRING, TensorShape({2, 2}));
  std::vector<Tensor> outputs = {classes, scores};
  EXPECT_CALL(*mock, Run(_, _, _, _, _, _, _))
      .WillRepeatedly(::testing::DoAll(::testing::SetArgPointee<4>(outputs),
                                       ::testing::Return(OkStatus())));
  TF_ASSERT_OK(Create());
  auto* examples =
      request_.mutable_input()->mutable_example_list()->mutable_examples();
  *examples->Add() = example({{"dos", 2}, {"uno", 1}});
  *examples->Add() = example({{"cuatro", 4}, {"tres", 3}});

  Status status = classifier_->Classify(request_, &result_);
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(),
              ::testing::HasSubstr("Expected scores Tensor of DT_FLOAT"));

  ClassificationResponse response;
  status = RunClassify(GetRunOptions(), saved_model_bundle_->meta_graph_def, {},
                       mock, request_, &response);
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(),
              ::testing::HasSubstr("Expected scores Tensor of DT_FLOAT"));
}

TEST_P(ClassifierTest, MismatchedNumberOfTensorClasses) {
  MockSession* mock = new MockSession;
  saved_model_bundle_->session.reset(mock);
  Tensor classes(DT_STRING, TensorShape({2, 2}));
  // Scores Tensor has three scores but classes only has two labels.
  Tensor scores(DT_FLOAT, TensorShape({2, 3}));
  std::vector<Tensor> outputs = {classes, scores};
  EXPECT_CALL(*mock, Run(_, _, _, _, _, _, _))
      .WillRepeatedly(::testing::DoAll(::testing::SetArgPointee<4>(outputs),
                                       ::testing::Return(OkStatus())));
  TF_ASSERT_OK(Create());
  auto* examples =
      request_.mutable_input()->mutable_example_list()->mutable_examples();
  *examples->Add() = example({{"dos", 2}, {"uno", 1}});
  *examples->Add() = example({{"cuatro", 4}, {"tres", 3}});

  Status status = classifier_->Classify(request_, &result_);
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(
      status.ToString(),
      ::testing::HasSubstr(
          "Tensors class and score should match in dim_size(1). Got 2 vs. 3"));

  ClassificationResponse response;
  status = RunClassify(GetRunOptions(), saved_model_bundle_->meta_graph_def, {},
                       mock, request_, &response);
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(),
              ::testing::HasSubstr("Tensors class and score should match in "
                                   "dim_size(1). Got 2 vs. 3"));
}

TEST_P(ClassifierTest, MethodNameCheck) {
  ClassificationResponse response;
  *request_.mutable_input()->mutable_example_list()->mutable_examples()->Add() =
      example({{"dos", 2}, {"uno", 1}});
  auto* signature_defs = meta_graph_def_->mutable_signature_def();

  // Legit method name. Should always work.
  (*signature_defs)[kDefaultServingSignatureDefKey].set_method_name(
      kClassifyMethodName);
  TF_EXPECT_OK(RunClassify(GetRunOptions(), *meta_graph_def_, {}, fake_session_,
                           request_, &response));

  // Unsupported method name will fail when method check is enabled.
  (*signature_defs)[kDefaultServingSignatureDefKey].set_method_name(
      "not/supported/method");
  EXPECT_EQ(RunClassify(GetRunOptions(), *meta_graph_def_, {}, fake_session_,
                        request_, &response)
                .ok(),
            !IsMethodNameCheckEnabled());

  // Empty method name will fail when method check is enabled.
  (*signature_defs)[kDefaultServingSignatureDefKey].clear_method_name();
  EXPECT_EQ(RunClassify(GetRunOptions(), *meta_graph_def_, {}, fake_session_,
                        request_, &response)
                .ok(),
            !IsMethodNameCheckEnabled());
}

INSTANTIATE_TEST_SUITE_P(Classifier, ClassifierTest, ::testing::Bool());

}  // namespace
}  // namespace serving
}  // namespace tensorflow
