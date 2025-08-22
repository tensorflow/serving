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

#include "tensorflow_serving/servables/tensorflow/saved_model_config_util.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/substitute.h"
#include "google/protobuf/text_format.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/grappler/optimizers/inference/batch_op_rewriter.pb.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow_serving/servables/tensorflow/remote_op_config_rewriter.pb.h"
#include "tensorflow_serving/servables/tensorflow/saved_model_config.pb.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace {

const char kTestSavedModelWithoutSavedModelConfigPath[] =
    "servables/tensorflow/testdata/"
    "saved_model_half_plus_two_cpu/00000123";

const char kTestSavedModelWithSavedModelConfigPath[] =
    "servables/tensorflow/testdata/"
    "saved_model_half_plus_two_cpu_with_saved_model_config/00000123";

const char kTestSavedModelWithEmptySavedModelConfigPath[] =
    "servables/tensorflow/testdata/"
    "saved_model_half_plus_two_cpu_with_empty_saved_model_config/00000123";

using test_util::EqualsProto;

TEST(LoadSavedModeConfigTest, MissingSavedModelConfig) {
  const std::string export_dir =
      test_util::TestSrcDirPath(kTestSavedModelWithoutSavedModelConfigPath);

  absl::StatusOr<SavedModelConfig> saved_model_config =
      LoadSavedModelConfigOrDefault(export_dir);
  TF_ASSERT_OK(saved_model_config.status());
  EXPECT_THAT(saved_model_config.value(), EqualsProto(""));
}

TEST(LoadSavedModelConfigTest, EmptySavedModelConfig) {
  const std::string export_dir =
      test_util::TestSrcDirPath(kTestSavedModelWithEmptySavedModelConfigPath);

  absl::StatusOr<SavedModelConfig> saved_model_config =
      LoadSavedModelConfigOrDefault(export_dir);

  TF_ASSERT_OK(saved_model_config.status());
  EXPECT_THAT(saved_model_config.value(), EqualsProto(""));
}

TEST(LoadSavedModelConfigTest, DISABLED_SavedModelConfig) {
  const std::string export_dir =
      test_util::TestSrcDirPath(kTestSavedModelWithSavedModelConfigPath);
  absl::StatusOr<SavedModelConfig> saved_model_config =
      LoadSavedModelConfigOrDefault(export_dir);

  TF_ASSERT_OK(saved_model_config.status());

  SavedModelConfig expected_config;
  bool result = ::google::protobuf::TextFormat::ParseFromString(
      R"pb(
        session_overrides {
          remote_op_remap_config {
            model_name_remap {
              key: "placeholder_model_name"
              value: "model_name"
            }
            target_address_remap {
              key: "placeholder_model_name"
              value: "target_address"
            }
          }
          batch_op_rewriter_config {
            batch_options {
              key: "placeholder_model_name"
              value: {
                batch_timeout_micros: 100
                allowed_batch_sizes: [ 2, 4, 8 ]
              }
            }
          }
        }
        tfrt_runtime_config {
          config {
            type_url: "type.googleapis.com/tensorflow.tfrt_stub.TestConfig1"
            value: "\n\rtest config 1"
          }
        }
        critical: true
      )pb",
      &expected_config);

  EXPECT_TRUE(result);
  EXPECT_THAT(saved_model_config.value(), EqualsProto(expected_config));
}

TEST(UpdateRewriterConfigTest, DISABLED_AddOptimizers) {
  const std::string export_dir =
      test_util::TestSrcDirPath(kTestSavedModelWithSavedModelConfigPath);
  absl::StatusOr<SavedModelConfig> saved_model_config =
      LoadSavedModelConfigOrDefault(export_dir);

  TF_ASSERT_OK(saved_model_config.status());
  tensorflow::RewriterConfig rewrite_options;

  UpdateRewriterConfig(saved_model_config.value().session_overrides(),
                       &rewrite_options);

  EXPECT_THAT(rewrite_options.custom_optimizers(),
              ::testing::UnorderedElementsAre(
                  EqualsProto(absl::Substitute(
                      R"pb(
                        name: "$0"
                        parameter_map {
                          key: "$1"
                          value { s: "$2" }
                        })pb",
                      kRemoteOpConfigRewriter, kRemoteOpRewriteConfigParamKey,
                      absl::Base64Escape(saved_model_config.value()
                                             .session_overrides()
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
                      absl::Base64Escape(saved_model_config.value()
                                             .session_overrides()
                                             .batch_op_rewriter_config()
                                             .SerializeAsString())))));
}

TEST(UpdateRewriterConfigTest, DISABLED_ReplaceOptimizers) {
  const std::string export_dir =
      test_util::TestSrcDirPath(kTestSavedModelWithSavedModelConfigPath);
  absl::StatusOr<SavedModelConfig> saved_model_config =
      LoadSavedModelConfigOrDefault(export_dir);

  TF_ASSERT_OK(saved_model_config.status());
  tensorflow::RewriterConfig rewrite_options;
  bool result = ::google::protobuf::TextFormat::ParseFromString(
      R"pb(
        custom_optimizers {
          name: "remote_op_config_rewrite"
          parameter_map {
            key: "remote_op_rewrite_config"
            value { s: "whatever placeholder value" }
          }
        }
        custom_optimizers {
          name: "batch_op_rewriter"
          parameter_map {
            key: "batch_op_rewrite_config"
            value { s: "whatever placeholder value" }
          }
        }
      )pb",
      &rewrite_options);

  UpdateRewriterConfig(saved_model_config.value().session_overrides(),
                       &rewrite_options);

  EXPECT_TRUE(result);
  EXPECT_THAT(rewrite_options.custom_optimizers(),
              ::testing::UnorderedElementsAre(
                  EqualsProto(absl::Substitute(
                      R"pb(
                        name: "$0"
                        parameter_map {
                          key: "$1"
                          value { s: "$2" }
                        })pb",
                      kRemoteOpConfigRewriter, kRemoteOpRewriteConfigParamKey,
                      absl::Base64Escape(saved_model_config.value()
                                             .session_overrides()
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
                      absl::Base64Escape(saved_model_config.value()
                                             .session_overrides()
                                             .batch_op_rewriter_config()
                                             .SerializeAsString())))));
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
