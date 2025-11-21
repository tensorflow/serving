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
#include "tensorflow_serving/servables/tensorflow/saved_model_config.h"

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
#include "tensorflow/core/tfrt/graph_executor/config.h"
#include "tensorflow/core/tfrt/graph_executor/test_config.pb.h"
#include "tsl/platform/path.h"
#include "tensorflow_serving/servables/tensorflow/remote_op_config_rewriter.pb.h"
#include "tensorflow_serving/servables/tensorflow/saved_model_config.pb.h"
#include "tensorflow_serving/servables/tensorflow/saved_model_config_util.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace {
const char kTestSavedModelWithoutSavedModelConfigPath[] =
    "servables/tensorflow/testdata/"
    "saved_model_half_plus_two_cpu/00000123";

const char kTestSavedModelWithModelConfigPath[] =
    "servables/tensorflow/testdata/"
    "saved_model_half_plus_two_cpu_with_saved_model_config/00000123";

const char kTestSavedModelWithEmptyModelConfigPath[] =
    "servables/tensorflow/testdata/"
    "saved_model_half_plus_two_cpu_with_empty_saved_model_config/00000123";

using test_util::EqualsProto;

TEST(SavedModeConfigTest, MissingSavedModelConfig) {
  const std::string export_dir =
      test_util::TestSrcDirPath(kTestSavedModelWithoutSavedModelConfigPath);
  tensorflow::GraphOptions graph_options;
  tensorflow::tfrt_stub::RuntimeConfig runtime_config;

  TF_ASSERT_OK(LoadSavedModelConfig(export_dir, graph_options, runtime_config));

  auto& custom_optimizers = graph_options.rewrite_options().custom_optimizers();
  EXPECT_EQ(custom_optimizers.size(), 0);
  EXPECT_EQ(runtime_config.ToProto().config_size(), 0);
}

TEST(ModelRuntimeConfigTest, EmptyModelConfig) {
  const std::string export_dir =
      test_util::TestSrcDirPath(kTestSavedModelWithEmptyModelConfigPath);
  tensorflow::GraphOptions graph_options;
  tensorflow::tfrt_stub::RuntimeConfig runtime_config;

  TF_ASSERT_OK(LoadSavedModelConfig(export_dir, graph_options, runtime_config));

  auto& custom_optimizers = graph_options.rewrite_options().custom_optimizers();
  EXPECT_EQ(custom_optimizers.size(), 0);
  EXPECT_EQ(runtime_config.ToProto().config_size(), 0);
}

TEST(ModelRuntimeConfigTest, DISABLED_OverwriteRuntimeConfig) {
  const std::string export_dir =
      test_util::TestSrcDirPath(kTestSavedModelWithModelConfigPath);
  tensorflow::GraphOptions graph_options;
  tensorflow::tfrt_stub::RuntimeConfig runtime_config;

  tensorflow::tfrt_stub::TestConfig1 old_test_config1;
  old_test_config1.set_tag("whatever tag");
  TF_ASSERT_OK(runtime_config.Add(old_test_config1));

  TF_ASSERT_OK(LoadSavedModelConfig(export_dir, graph_options, runtime_config));

  auto& custom_optimizers = graph_options.rewrite_options().custom_optimizers();
  EXPECT_EQ(custom_optimizers.size(), 2);
  EXPECT_THAT(
      runtime_config.ToProto(), EqualsProto(R"pb(
        config {
          type_url: "type.googleapis.com/tensorflow.tfrt_stub.TestConfig1"
          value: "\n\rtest config 1"
        }
      )pb"));
}

TEST(ModelRuntimeConfigTest, DISABLED_ModelConfig) {
  const std::string export_dir =
      test_util::TestSrcDirPath(kTestSavedModelWithModelConfigPath);
  SavedModelConfig model_config;
  {
    std::string content;
    TF_ASSERT_OK(tsl::ReadFileToString(
        tsl::Env::Default(),
        test_util::TestSrcDirPath(tsl::io::JoinPath(
            kTestSavedModelWithModelConfigPath, kSavedModelAssetsExtraDirectory,
            kSavedModelConfigPath)),
        &content));

    EXPECT_TRUE(model_config.ParseFromString(content));
  }

  tensorflow::GraphOptions graph_options;
  tensorflow::tfrt_stub::RuntimeConfig runtime_config;

  TF_ASSERT_OK(LoadSavedModelConfig(export_dir, graph_options, runtime_config));

  auto& custom_optimizers = graph_options.rewrite_options().custom_optimizers();
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
                      absl::Base64Escape(model_config.session_overrides()
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
                      absl::Base64Escape(model_config.session_overrides()
                                             .batch_op_rewriter_config()
                                             .SerializeAsString())))));
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
