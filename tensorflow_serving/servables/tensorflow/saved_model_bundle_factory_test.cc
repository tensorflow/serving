/* Copyright 2016 Google Inc. All Rights Reserved.

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

#include "tensorflow_serving/servables/tensorflow/saved_model_bundle_factory.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/wrappers.pb.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/protobuf/named_tensor.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow_serving/core/test_util/session_test_util.h"
#include "tensorflow_serving/servables/tensorflow/bundle_factory_test.h"
#include "tensorflow_serving/servables/tensorflow/bundle_factory_test_util.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_config.pb.h"

namespace tensorflow {
namespace serving {
namespace {

enum class CreationType { kWithoutMetadata, kWithMetadata };

enum class ModelType { kTfModel, kTfLiteModel };

Loader::Metadata CreateMetadata() { return {ServableId{"name", 42}}; }

// Creates a new session based on the config and export path.
Status CreateBundleFromPath(const CreationType creation_type,
                            const SessionBundleConfig& config,
                            const string& path,
                            std::unique_ptr<SavedModelBundle>* bundle) {
  std::unique_ptr<SavedModelBundleFactory> factory;
  TF_RETURN_IF_ERROR(SavedModelBundleFactory::Create(config, &factory));
  auto config_with_session_hook = config;
  config_with_session_hook.set_session_target(
      test_util::kNewSessionHookSessionTargetPrefix);
  test_util::SetNewSessionHook([&](const SessionOptions& session_options) {
    const bool enable_session_metadata =
        creation_type == CreationType::kWithMetadata;
    EXPECT_EQ(enable_session_metadata,
              session_options.config.experimental().has_session_metadata());
    if (enable_session_metadata) {
      const auto& actual_session_metadata =
          session_options.config.experimental().session_metadata();
      const auto& expected_loader_metadata = CreateMetadata();
      EXPECT_EQ(expected_loader_metadata.servable_id.name,
                actual_session_metadata.name());
      EXPECT_EQ(expected_loader_metadata.servable_id.version,
                actual_session_metadata.version());
    }
    return Status::OK();
  });

  switch (creation_type) {
    case CreationType::kWithoutMetadata:
      TF_RETURN_IF_ERROR(factory->CreateSavedModelBundle(path, bundle));
      break;
    case CreationType::kWithMetadata:
      TF_RETURN_IF_ERROR(factory->CreateSavedModelBundleWithMetadata(
          CreateMetadata(), path, bundle));
      break;
  }
  return Status::OK();
}

// Tests SavedModelBundleFactory with native SavedModel.
class SavedModelBundleFactoryTest
    : public test_util::BundleFactoryTest,
      public ::testing::WithParamInterface<std::pair<CreationType, ModelType>> {
 public:
  SavedModelBundleFactoryTest()
      : test_util::BundleFactoryTest(
            GetParam().second == ModelType::kTfModel
                ? test_util::GetTestSavedModelPath()
                : test_util::GetTestTfLiteModelPath()) {}

  virtual ~SavedModelBundleFactoryTest() = default;

 protected:
  Status CreateSession(const SessionBundleConfig& config,
                       std::unique_ptr<Session>* session) const override {
    std::unique_ptr<SavedModelBundle> bundle;
    TF_RETURN_IF_ERROR(
        CreateBundleFromPath(GetParam().first, config, export_dir_, &bundle));
    *session = std::move(bundle->session);
    return Status::OK();
  }

  SessionBundleConfig GetSessionBundleConfig() const override {
    SessionBundleConfig config;
    if (GetParam().second == ModelType::kTfLiteModel) {
      config.set_use_tflite_model(true);
    }
    return config;
  }

  bool IsRunOptionsSupported() const override {
    // Presently TensorFlow Lite sessions do NOT support RunOptions.
    return GetParam().second != ModelType::kTfLiteModel;
  }

  std::vector<string> GetModelFiles() {
    switch (GetParam().second) {
      case ModelType::kTfModel: {
        const string& dir = test_util::GetTestSavedModelPath();
        return {
            io::JoinPath(dir, kSavedModelAssetsDirectory, "foo.txt"),
            io::JoinPath(dir, kSavedModelFilenamePb),
            io::JoinPath(dir, kSavedModelVariablesFilename,
                         "variables.data-00000-of-00001"),
            io::JoinPath(dir, kSavedModelVariablesFilename, "variables.index")};
      }
      case ModelType::kTfLiteModel: {
        return {
            io::JoinPath(test_util::GetTestTfLiteModelPath(), "model.tflite")};
      }
      default:
        return {};
    }
  }
};

INSTANTIATE_TEST_SUITE_P(
    CreationType, SavedModelBundleFactoryTest,
    ::testing::Values(
        std::make_pair(CreationType::kWithoutMetadata, ModelType::kTfModel),
        std::make_pair(CreationType::kWithoutMetadata, ModelType::kTfLiteModel),
        std::make_pair(CreationType::kWithMetadata, ModelType::kTfModel),
        std::make_pair(CreationType::kWithMetadata, ModelType::kTfLiteModel)));

TEST_P(SavedModelBundleFactoryTest, Basic) { TestBasic(); }

TEST_P(SavedModelBundleFactoryTest, FixedInputTensors) {
  Tensor fixed_input = test::AsTensor<float>({100.0f, 42.0f}, {2});
  NamedTensorProto fixed_input_proto;
  fixed_input_proto.set_name("x:0");
  fixed_input.AsProtoField(fixed_input_proto.mutable_tensor());

  SessionBundleConfig config = GetSessionBundleConfig();
  *config.add_saved_model_tags() = kSavedModelTagServe;
  *config.add_experimental_fixed_input_tensors() = fixed_input_proto;
  std::unique_ptr<Session> session;
  TF_ASSERT_OK(CreateSession(config, &session));

  // half plus two: output should be input / 2 + 2.
  const Tensor expected_output =
      test::AsTensor<float>({100.0f / 2 + 2, 42.0f / 2 + 2}, {2});

  const std::vector<std::pair<string, Tensor>> non_fixed_inputs = {};
  const std::vector<string> output_names = {"y:0"};
  const std::vector<string> empty_targets;
  std::vector<Tensor> outputs;
  TF_ASSERT_OK(
      session->Run(non_fixed_inputs, output_names, empty_targets, &outputs));
  ASSERT_EQ(1, outputs.size());
  const Tensor& single_output = outputs.at(0);
  test::ExpectTensorEqual<float>(expected_output, single_output);
}

TEST_P(SavedModelBundleFactoryTest, RemoveUnusedFieldsFromMetaGraphDefault) {
  SessionBundleConfig config = GetSessionBundleConfig();
  *config.add_saved_model_tags() = kSavedModelTagServe;
  std::unique_ptr<SavedModelBundle> bundle;
  TF_ASSERT_OK(
      CreateBundleFromPath(GetParam().first, config, export_dir_, &bundle));
  if (GetParam().second == ModelType::kTfLiteModel) {
    // TF Lite model never has a graph_def.
    EXPECT_FALSE(bundle->meta_graph_def.has_graph_def());
  } else {
    EXPECT_TRUE(bundle->meta_graph_def.has_graph_def());
  }
  EXPECT_FALSE(bundle->meta_graph_def.signature_def().empty());
}

TEST_P(SavedModelBundleFactoryTest, RemoveUnusedFieldsFromMetaGraphEnabled) {
  SessionBundleConfig config = GetSessionBundleConfig();
  *config.add_saved_model_tags() = kSavedModelTagServe;
  config.set_remove_unused_fields_from_bundle_metagraph(true);
  std::unique_ptr<SavedModelBundle> bundle;
  TF_ASSERT_OK(
      CreateBundleFromPath(GetParam().first, config, export_dir_, &bundle));
  EXPECT_FALSE(bundle->meta_graph_def.has_graph_def());
  EXPECT_FALSE(bundle->meta_graph_def.signature_def().empty());
}

TEST_P(SavedModelBundleFactoryTest, Batching) { TestBatching(); }

TEST_P(SavedModelBundleFactoryTest, EstimateResourceRequirementWithGoodExport) {
  const double kTotalFileSize = test_util::GetTotalFileSize(GetModelFiles());
  TestEstimateResourceRequirementWithGoodExport<SavedModelBundleFactory>(
      kTotalFileSize);
}

TEST_P(SavedModelBundleFactoryTest, RunOptions) { TestRunOptions(); }

TEST_P(SavedModelBundleFactoryTest, RunOptionsError) { TestRunOptionsError(); }

// Tests SavedModelBundleFactory with SessionBundle export.
class SavedModelBundleFactoryBackwardCompatibilityTest
    : public test_util::BundleFactoryTest,
      public ::testing::WithParamInterface<CreationType> {
 public:
  SavedModelBundleFactoryBackwardCompatibilityTest()
      : test_util::BundleFactoryTest(
            test_util::GetTestSessionBundleExportPath()) {}

  virtual ~SavedModelBundleFactoryBackwardCompatibilityTest() = default;

 private:
  Status CreateSession(const SessionBundleConfig& config,
                       std::unique_ptr<Session>* session) const override {
    std::unique_ptr<SavedModelBundle> bundle;
    TF_RETURN_IF_ERROR(
        CreateBundleFromPath(GetParam(), config, export_dir_, &bundle));
    *session = std::move(bundle->session);
    return Status::OK();
  }
};

INSTANTIATE_TEST_SUITE_P(CreationType,
                         SavedModelBundleFactoryBackwardCompatibilityTest,
                         ::testing::Values(CreationType::kWithoutMetadata,
                                           CreationType::kWithMetadata));

TEST_P(SavedModelBundleFactoryBackwardCompatibilityTest, Basic) { TestBasic(); }

TEST_P(SavedModelBundleFactoryBackwardCompatibilityTest, Batching) {
  TestBatching();
}

TEST_P(SavedModelBundleFactoryBackwardCompatibilityTest,
       EstimateResourceRequirementWithGoodExport) {
  const double kTotalFileSize =
      test_util::GetTotalFileSize(test_util::GetTestSessionBundleExportFiles());
  TestEstimateResourceRequirementWithGoodExport<SavedModelBundleFactory>(
      kTotalFileSize);
}

TEST_P(SavedModelBundleFactoryBackwardCompatibilityTest, RunOptions) {
  TestRunOptions();
}

TEST_P(SavedModelBundleFactoryBackwardCompatibilityTest, RunOptionsError) {
  TestRunOptionsError();
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
