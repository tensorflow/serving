/* Copyright 2023 Google Inc. All Rights Reserved.

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

#include "tensorflow_serving/session_bundle/saved_model_config.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/escaping.h"
#include "absl/strings/substitute.h"
#include "tensorflow/cc/saved_model/constants.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/status.h"
#include "tensorflow/core/grappler/optimizers/inference/batch_op_rewriter.pb.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tsl/platform/path.h"
#include "tensorflow_serving/servables/tensorflow/remote_op_config_rewriter.pb.h"
#include "tensorflow_serving/servables/tensorflow/saved_model_config.pb.h"
#include "tensorflow_serving/servables/tensorflow/saved_model_config_util.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace session_bundle {
namespace {

const char kTestSavedModelWithSavedModelConfigPath[] =
    "servables/tensorflow/testdata/"
    "saved_model_half_plus_two_cpu_with_saved_model_config/00000123";

const char kTestSavedModelWithEmptySavedModelConfigPath[] =
    "servables/tensorflow/testdata/"
    "saved_model_half_plus_two_cpu_with_empty_saved_model_config/00000123";

using test_util::EqualsProto;

TEST(SavedModelConfigTest, EmptySavedModelConfig) {
  const std::string export_dir =
      test_util::TestSrcDirPath(kTestSavedModelWithEmptySavedModelConfigPath);

  SessionOptions session_options;

  TF_ASSERT_OK(MaybeLoadSavedModelConfig(export_dir, &session_options));

  auto& custom_optimizers = session_options.config.graph_options()
                                .rewrite_options()
                                .custom_optimizers();
  EXPECT_EQ(custom_optimizers.size(), 0);
}

TEST(SavedModelConfigTest, DISABLED_SavedModelConfig) {
  const std::string export_dir =
      test_util::TestSrcDirPath(kTestSavedModelWithSavedModelConfigPath);

  SessionOptions session_options;

  SavedModelConfig saved_model_config;
  {
    std::string content;
    TF_ASSERT_OK(tsl::ReadFileToString(
        tsl::Env::Default(),
        test_util::TestSrcDirPath(tsl::io::JoinPath(
            kTestSavedModelWithSavedModelConfigPath,
            kSavedModelAssetsExtraDirectory, kSavedModelConfigPath)),
        &content));

    EXPECT_TRUE(saved_model_config.ParseFromString(content));
  }

  TF_ASSERT_OK(MaybeLoadSavedModelConfig(export_dir, &session_options));

  auto& custom_optimizers = session_options.config.graph_options()
                                .rewrite_options()
                                .custom_optimizers();
  EXPECT_EQ(custom_optimizers.size(), 2);

  EXPECT_THAT(custom_optimizers,
              ::testing::UnorderedElementsAre(
                  EqualsProto(absl::Substitute(
                      R"pb(
                        name: "$0"
                        parameter_map {
                          key: "$1"
                          value { s: "$2" }
                        })pb",
                      kRemoteOpConfigRewriter, kRemoteOpRewriteConfigParamKey,
                      absl::Base64Escape(saved_model_config.session_overrides()
                                             .remote_op_remap_config()
                                             .SerializeAsString()))),
                  EqualsProto(absl::Substitute(
                      R"pb(
                        name: "$0"
                        parameter_map {
                          key: "$1"
                          value { s: "$2" }
                        })pb",
                      kBatchOpRewriter, kBatchOpRewriteConfigParamKey,
                      absl::Base64Escape(saved_model_config.session_overrides()
                                             .batch_op_rewriter_config()
                                             .SerializeAsString())))));
}

}  // namespace
}  // namespace session_bundle
}  // namespace serving
}  // namespace tensorflow
