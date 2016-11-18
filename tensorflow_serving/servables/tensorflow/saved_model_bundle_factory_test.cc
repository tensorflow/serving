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
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow_serving/servables/tensorflow/bundle_factory_test.h"
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

 private:
  Status CreateSession(const SessionBundleConfig& config,
                       std::unique_ptr<Session>* session) const override {
    return CreateSessionFromPath(config, export_dir_, session);
  }
};

TEST_F(SavedModelBundleFactoryTest, Basic) { TestBasic(); }

TEST_F(SavedModelBundleFactoryTest, Batching) { TestBatching(); }

TEST_F(SavedModelBundleFactoryTest, EstimateResourceRequirementWithGoodExport) {
  const double kTotalFileSize = 7492;
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
  // The length of the file's version strings might change, so we don't
  // hardcode their size.  They are 4 bytes for tags & size, plus the actual
  // length of the strings.
  const double kVersionSize =
      4 + strlen(TF_VERSION_STRING) + strlen(tf_git_version());
  const double kTotalFileSize = 13392.5 + kVersionSize;
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
