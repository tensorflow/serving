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
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/protobuf/named_tensor.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow_serving/servables/tensorflow/bundle_factory_test.h"
#include "tensorflow_serving/servables/tensorflow/bundle_factory_test_util.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_config.pb.h"

namespace tensorflow {
namespace serving {
namespace {

// Creates a new session based on the config and export path.
Status CreateSessionFromPath(const SessionBundleConfig& config,
                             const string& path,
                             std::unique_ptr<Session>* session) {
  std::unique_ptr<SavedModelBundleFactory> factory;
  TF_RETURN_IF_ERROR(SavedModelBundleFactory::Create(config, &factory));
  std::unique_ptr<SavedModelBundle> bundle;
  TF_RETURN_IF_ERROR(factory->CreateSavedModelBundle(path, &bundle));
  *session = std::move(bundle->session);
  return Status::OK();
}

// Tests SavedModelBundleFactory with native SavedModel.
class SavedModelBundleFactoryTest : public test_util::BundleFactoryTest {
 public:
  SavedModelBundleFactoryTest()
      : test_util::BundleFactoryTest(test_util::GetTestSavedModelPath()) {}

  virtual ~SavedModelBundleFactoryTest() = default;

 protected:
  Status CreateSession(const SessionBundleConfig& config,
                       std::unique_ptr<Session>* session) const override {
    return CreateSessionFromPath(config, export_dir_, session);
  }
};

TEST_F(SavedModelBundleFactoryTest, Basic) { TestBasic(); }

TEST_F(SavedModelBundleFactoryTest, FixedInputTensors) {
  Tensor fixed_input = test::AsTensor<float>({100.0f, 42.0f}, {2});
  NamedTensorProto fixed_input_proto;
  fixed_input_proto.set_name("x:0");
  fixed_input.AsProtoField(fixed_input_proto.mutable_tensor());

  SessionBundleConfig config;
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

TEST_F(SavedModelBundleFactoryTest, Batching) { TestBatching(); }

TEST_F(SavedModelBundleFactoryTest, EstimateResourceRequirementWithGoodExport) {
  const double kTotalFileSize =
      test_util::GetTotalFileSize(test_util::GetTestSavedModelFiles());
  TestEstimateResourceRequirementWithGoodExport<SavedModelBundleFactory>(
      kTotalFileSize);
}

TEST_F(SavedModelBundleFactoryTest, RunOptions) { TestRunOptions(); }

TEST_F(SavedModelBundleFactoryTest, RunOptionsError) { TestRunOptionsError(); }

// Tests SavedModelBundleFactory with SessionBundle export.
class SavedModelBundleFactoryBackwardCompatibilityTest
    : public test_util::BundleFactoryTest {
 public:
  SavedModelBundleFactoryBackwardCompatibilityTest()
      : test_util::BundleFactoryTest(
            test_util::GetTestSessionBundleExportPath()) {}

  virtual ~SavedModelBundleFactoryBackwardCompatibilityTest() = default;

 private:
  Status CreateSession(const SessionBundleConfig& config,
                       std::unique_ptr<Session>* session) const override {
    return CreateSessionFromPath(config, export_dir_, session);
  }
};

TEST_F(SavedModelBundleFactoryBackwardCompatibilityTest, Basic) { TestBasic(); }

TEST_F(SavedModelBundleFactoryBackwardCompatibilityTest, Batching) {
  TestBatching();
}

TEST_F(SavedModelBundleFactoryBackwardCompatibilityTest,
       EstimateResourceRequirementWithGoodExport) {
  const double kTotalFileSize =
      test_util::GetTotalFileSize(test_util::GetTestSessionBundleExportFiles());
  TestEstimateResourceRequirementWithGoodExport<SavedModelBundleFactory>(
      kTotalFileSize);
}

TEST_F(SavedModelBundleFactoryBackwardCompatibilityTest, RunOptions) {
  TestRunOptions();
}

TEST_F(SavedModelBundleFactoryBackwardCompatibilityTest, RunOptionsError) {
  TestRunOptionsError();
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
