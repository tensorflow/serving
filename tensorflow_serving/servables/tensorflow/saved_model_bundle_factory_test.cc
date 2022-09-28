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

#include <fstream>
#include <iostream>
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
    return OkStatus();
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
  return OkStatus();
}

struct SavedModelBundleFactoryTestParam {
  CreationType creation_type;
  ModelType model_type;
  bool prefer_tflite_model;
};

// Tests SavedModelBundleFactory with native SavedModel.
class SavedModelBundleFactoryTest
    : public test_util::BundleFactoryTest,
      public ::testing::WithParamInterface<SavedModelBundleFactoryTestParam> {
 public:
  SavedModelBundleFactoryTest()
      : test_util::BundleFactoryTest(
            GetParam().model_type == ModelType::kTfModel
                ? test_util::GetTestSavedModelPath()
                : test_util::GetTestTfLiteModelPath()) {}

  virtual ~SavedModelBundleFactoryTest() = default;

 protected:
  Status CreateSession(const SessionBundleConfig& config,
                       std::unique_ptr<Session>* session) const override {
    std::unique_ptr<SavedModelBundle> bundle;
    TF_RETURN_IF_ERROR(CreateBundleFromPath(GetParam().creation_type, config,
                                            export_dir_, &bundle));
    *session = std::move(bundle->session);
    return OkStatus();
  }

  SessionBundleConfig GetSessionBundleConfig() const override {
    SessionBundleConfig config;
    config.set_prefer_tflite_model(GetParam().prefer_tflite_model);
    return config;
  }

  bool IsRunOptionsSupported() const override {
    // Presently TensorFlow Lite sessions do NOT support RunOptions.
    return GetParam().prefer_tflite_model == false ||
           GetParam().model_type != ModelType::kTfLiteModel;
  }

  bool ExpectCreateBundleFailure() const override {
    // The test Tensorflow Lite model does not include saved_model artifacts
    return GetParam().prefer_tflite_model == false &&
           GetParam().model_type == ModelType::kTfLiteModel;
  }

  std::vector<string> GetModelFiles() {
    switch (GetParam().model_type) {
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
        SavedModelBundleFactoryTestParam{
            CreationType::kWithoutMetadata, ModelType::kTfModel,
            true  // prefer_tflite_model
        },
        SavedModelBundleFactoryTestParam{
            CreationType::kWithoutMetadata, ModelType::kTfModel,
            false  // prefer_tflite_model
        },
        SavedModelBundleFactoryTestParam{
            CreationType::kWithoutMetadata, ModelType::kTfLiteModel,
            true  // prefer_tflite_model
        },
        SavedModelBundleFactoryTestParam{
            CreationType::kWithoutMetadata, ModelType::kTfLiteModel,
            false  // prefer_tflite_model
        },
        SavedModelBundleFactoryTestParam{
            CreationType::kWithMetadata, ModelType::kTfModel,
            true  // prefer_tflite_model
        },
        SavedModelBundleFactoryTestParam{
            CreationType::kWithMetadata, ModelType::kTfModel,
            false  // prefer_tflite_model
        },
        SavedModelBundleFactoryTestParam{
            CreationType::kWithMetadata, ModelType::kTfLiteModel,
            true  // prefer_tflite_model
        },
        SavedModelBundleFactoryTestParam{
            CreationType::kWithMetadata, ModelType::kTfLiteModel,
            false  // prefer_tflite_model
        }));

TEST_P(SavedModelBundleFactoryTest, Basic) { TestBasic(); }

TEST_P(SavedModelBundleFactoryTest, RemoveUnusedFieldsFromMetaGraphDefault) {
  SessionBundleConfig config = GetSessionBundleConfig();
  *config.add_saved_model_tags() = kSavedModelTagServe;
  std::unique_ptr<SavedModelBundle> bundle;
  if (ExpectCreateBundleFailure()) {
    EXPECT_FALSE(CreateBundleFromPath(GetParam().creation_type, config,
                                      export_dir_, &bundle)
                     .ok());
    return;
  }
  TF_ASSERT_OK(CreateBundleFromPath(GetParam().creation_type, config,
                                    export_dir_, &bundle));
  if (GetParam().prefer_tflite_model &&
      (GetParam().model_type == ModelType::kTfLiteModel)) {
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
  if (ExpectCreateBundleFailure()) {
    EXPECT_FALSE(CreateBundleFromPath(GetParam().creation_type, config,
                                      export_dir_, &bundle)
                     .ok());
    return;
  }
  TF_ASSERT_OK(CreateBundleFromPath(GetParam().creation_type, config,
                                    export_dir_, &bundle));
  EXPECT_FALSE(bundle->meta_graph_def.has_graph_def());
  EXPECT_FALSE(bundle->meta_graph_def.signature_def().empty());
}

TEST_P(SavedModelBundleFactoryTest, Batching) {
  // Most test cases don't cover batching session code path so call
  // 'TestBatching' twice with different options for batching test case, as
  // opposed to parameterize test.
  TestBatching(test_util::CreateProto<BatchingParameters>(R"(
    max_batch_size { value: 4 }
    enable_large_batch_splitting { value: False })"),
               /*enable_per_model_batching_params=*/false,
               /*input_request_batch_size=*/2,
               /*batch_size=*/4);

  TestBatching(test_util::CreateProto<BatchingParameters>(R"(
    max_batch_size { value: 4 }
    enable_large_batch_splitting { value: True }
    max_execution_batch_size { value: 2 })"),
               /*enable_per_model_batching_params=*/false,
               /*input_request_batch_size=*/3,
               /*batch_size=*/2);
}

TEST_P(SavedModelBundleFactoryTest, PerModelBatchingParams) {
  //
  // Copy SavedModel to temp (writable) location, and add batching params.
  //
  const string dst_dir = io::JoinPath(testing::TmpDir(), "model");
  test_util::CopyDirOrDie(export_dir_, dst_dir);
  // Note, timeout is set to high value to force batch formation.
  const string& per_model_params_pbtxt(R"(
    max_batch_size { value: 10 }
    batch_timeout_micros { value: 100000000 })");
  std::ofstream ofs(io::JoinPath(dst_dir, "batching_params.pbtxt"));
  ofs << per_model_params_pbtxt;
  ofs.close();
  export_dir_ = dst_dir;

  const BatchingParameters& common_params =
      test_util::CreateProto<BatchingParameters>(
          R"(max_batch_size { value: 4 })");
  TestBatching(common_params, /*enable_per_model_batching_params=*/false,
               /*input_request_batch_size=*/2, /*batch_size=*/4);
  TestBatching(common_params, /*enable_per_model_batching_params=*/true,
               /*input_request_batch_size=*/2, /*batch_size=*/10);
}

TEST_P(SavedModelBundleFactoryTest, EstimateResourceRequirementWithGoodExport) {
  const double kTotalFileSize = test_util::GetTotalFileSize(GetModelFiles());
  TestEstimateResourceRequirementWithGoodExport<SavedModelBundleFactory>(
      kTotalFileSize);
}

TEST_P(SavedModelBundleFactoryTest, RunOptions) { TestRunOptions(); }

TEST_P(SavedModelBundleFactoryTest, RunOptionsError) { TestRunOptionsError(); }

}  // namespace
}  // namespace serving
}  // namespace tensorflow
