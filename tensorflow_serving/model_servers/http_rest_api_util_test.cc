/* Copyright 2021 Google Inc. All Rights Reserved.

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

#include "tensorflow_serving/model_servers/http_rest_api_util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace {

using ::testing::HasSubstr;

class HttpRestApiUtilTest : public ::testing::Test {};

TEST_F(HttpRestApiUtilTest, TestParseModelInfoForGet) {
  string model_name;
  absl::optional<int64_t> model_version;
  absl::optional<string> model_version_label;
  string method;
  string model_subresource;
  bool parse_successful;

  // Test reserved charactors in model name.
  const absl::string_view request_path_1 =
      "/v1/models/"
      "modelname%21%23%24%25%26%27%28%29%2A%2B%2C%2F%3A%3B%3D%3F%40%5B%5D";
  TF_EXPECT_OK(ParseModelInfo("GET", request_path_1, &model_name,
                              &model_version, &model_version_label, &method,
                              &model_subresource, &parse_successful));
  EXPECT_EQ(model_name, "modelname!#$%&'()*+,/:;=?@[]");
  EXPECT_TRUE(parse_successful);

  // Test model version.
  const absl::string_view request_path_2 = "/v1/models/modelname/versions/99";
  TF_EXPECT_OK(ParseModelInfo("GET", request_path_2, &model_name,
                              &model_version, &model_version_label, &method,
                              &model_subresource, &parse_successful));
  EXPECT_EQ(model_name, "modelname");
  EXPECT_EQ(model_version.value(), 99);
  EXPECT_TRUE(parse_successful);

  // Test labels.
  const absl::string_view request_path_3 = "/v1/models/modelname/labels/latest";
  TF_EXPECT_OK(ParseModelInfo("GET", request_path_3, &model_name,
                              &model_version, &model_version_label, &method,
                              &model_subresource, &parse_successful));
  EXPECT_EQ(model_name, "modelname");
  EXPECT_EQ(model_version_label.value(), "latest");
  EXPECT_TRUE(parse_successful);

  // Test labels with reserved charactors.
  const absl::string_view request_path_4 =
      "/v1/models/modelname/labels/"
      "latest%21%23%24%25%26%27%28%29%2A%2B%2C%2F%3A%3B%3D%3F%40%5B%5D";
  TF_EXPECT_OK(ParseModelInfo("GET", request_path_4, &model_name,
                              &model_version, &model_version_label, &method,
                              &model_subresource, &parse_successful));
  EXPECT_EQ(model_name, "modelname");
  EXPECT_EQ(model_version_label.value(), "latest!#$%&'()*+,/:;=?@[]");
  EXPECT_TRUE(parse_successful);

  // Test metadata.
  const absl::string_view request_path_5 =
      "/v1/models/modelname/labels/latest/metadata";
  TF_EXPECT_OK(ParseModelInfo("GET", request_path_5, &model_name,
                              &model_version, &model_version_label, &method,
                              &model_subresource, &parse_successful));
  EXPECT_EQ(model_name, "modelname");
  EXPECT_EQ(model_version_label.value(), "latest");
  EXPECT_EQ(model_subresource, "metadata");
  EXPECT_TRUE(parse_successful);

  // Test metadata with reserved charactors.
  const absl::string_view request_path_6 =
      "/v1/models/"
      "modelname%21%23%24%25%26%27%28%29%2A%2B%2C%2F%3A%3B%3D%3F%40%5B%5D/"
      "labels/latest%21%23%24%25%26%27%28%29%2A%2B%2C%2F%3A%3B%3D%3F%40%5B%5D/"
      "metadata";
  TF_EXPECT_OK(ParseModelInfo("GET", request_path_6, &model_name,
                              &model_version, &model_version_label, &method,
                              &model_subresource, &parse_successful));
  EXPECT_EQ(model_name, "modelname!#$%&'()*+,/:;=?@[]");
  EXPECT_EQ(model_version_label.value(), "latest!#$%&'()*+,/:;=?@[]");
  EXPECT_EQ(model_subresource, "metadata");
  EXPECT_TRUE(parse_successful);

  // Test failure cases.
  TF_EXPECT_OK(ParseModelInfo("GET", "/v1/foo", &model_name, &model_version,
                              &model_version_label, &method, &model_subresource,
                              &parse_successful));
  EXPECT_FALSE(parse_successful);
  TF_EXPECT_OK(ParseModelInfo("GET", "/v1/models/foo:predict", &model_name,
                              &model_version, &model_version_label, &method,
                              &model_subresource, &parse_successful));
  EXPECT_FALSE(parse_successful);
  TF_EXPECT_OK(ParseModelInfo("GET", "/v1/models/foo/version/50:predict",
                              &model_name, &model_version, &model_version_label,
                              &method, &model_subresource, &parse_successful));
  EXPECT_FALSE(parse_successful);
}

TEST_F(HttpRestApiUtilTest, TestParseModelInfoForPost) {
  string model_name;
  absl::optional<int64_t> model_version;
  absl::optional<string> model_version_label;
  string method;
  string model_subresource;
  bool parse_successful;

  // Test reserved charactors in model name.
  const absl::string_view request_path_1 =
      "/v1/models/"
      "modelname%21%23%24%25%26%27%28%29%2A%2B%2C%2F%3A%3B%3D%3F%40%5B%5D:"
      "predict";
  TF_EXPECT_OK(ParseModelInfo("POST", request_path_1, &model_name,
                              &model_version, &model_version_label, &method,
                              &model_subresource, &parse_successful));
  EXPECT_EQ(model_name, "modelname!#$%&'()*+,/:;=?@[]");
  EXPECT_EQ(method, "predict");
  EXPECT_TRUE(parse_successful);

  // Test model version.
  const absl::string_view request_path_2 =
      "/v1/models/modelname/versions/99:predict";
  TF_EXPECT_OK(ParseModelInfo("POST", request_path_2, &model_name,
                              &model_version, &model_version_label, &method,
                              &model_subresource, &parse_successful));
  EXPECT_EQ(model_name, "modelname");
  EXPECT_EQ(model_version.value(), 99);
  EXPECT_EQ(method, "predict");
  EXPECT_TRUE(parse_successful);

  // Test labels.
  const absl::string_view request_path_3 =
      "/v1/models/modelname/labels/latest:predict";
  TF_EXPECT_OK(ParseModelInfo("POST", request_path_3, &model_name,
                              &model_version, &model_version_label, &method,
                              &model_subresource, &parse_successful));
  EXPECT_EQ(model_name, "modelname");
  EXPECT_EQ(model_version_label.value(), "latest");
  EXPECT_EQ(method, "predict");
  EXPECT_TRUE(parse_successful);

  // Test labels with reserved charactors.
  const absl::string_view request_path_4 =
      "/v1/models/modelname/labels/"
      "latest%21%23%24%25%26%27%28%29%2A%2B%2C%2F%3A%3B%3D%3F%40%5B%5D:predict";
  TF_EXPECT_OK(ParseModelInfo("POST", request_path_4, &model_name,
                              &model_version, &model_version_label, &method,
                              &model_subresource, &parse_successful));
  EXPECT_EQ(model_name, "modelname");
  EXPECT_EQ(model_version_label.value(), "latest!#$%&'()*+,/:;=?@[]");
  EXPECT_EQ(method, "predict");
  EXPECT_TRUE(parse_successful);

  // Test reserved charactors in labels and model name.
  const absl::string_view request_path_6 =
      "/v1/models/"
      "modelname%21%23%24%25%26%27%28%29%2A%2B%2C%2F%3A%3B%3D%3F%40%5B%5D/"
      "labels/"
      "latest%21%23%24%25%26%27%28%29%2A%2B%2C%2F%3A%3B%3D%3F%40%5B%5D:predict";
  TF_EXPECT_OK(ParseModelInfo("POST", request_path_6, &model_name,
                              &model_version, &model_version_label, &method,
                              &model_subresource, &parse_successful));
  EXPECT_EQ(model_name, "modelname!#$%&'()*+,/:;=?@[]");
  EXPECT_EQ(model_version_label.value(), "latest!#$%&'()*+,/:;=?@[]");
  EXPECT_EQ(method, "predict");
  EXPECT_TRUE(parse_successful);

  // Test failure cases.
  TF_EXPECT_OK(ParseModelInfo("POST", "/v1/foo", &model_name, &model_version,
                              &model_version_label, &method, &model_subresource,
                              &parse_successful));
  EXPECT_FALSE(parse_successful);
  TF_EXPECT_OK(ParseModelInfo("POST", "/v1/models", &model_name, &model_version,
                              &model_version_label, &method, &model_subresource,
                              &parse_successful));
  EXPECT_FALSE(parse_successful);
  TF_EXPECT_OK(ParseModelInfo("POST", "/v1/models/foo/version/50:predict",
                              &model_name, &model_version, &model_version_label,
                              &method, &model_subresource, &parse_successful));
  EXPECT_FALSE(parse_successful);
  TF_EXPECT_OK(ParseModelInfo("POST", "/v1/models/foo/versions/string:predict",
                              &model_name, &model_version, &model_version_label,
                              &method, &model_subresource, &parse_successful));
  EXPECT_FALSE(parse_successful);
  auto status = ParseModelInfo(
      "POST",
      absl::StrCat("/v1/models/foo/versions/",
                   std::numeric_limits<uint64_t>::max(), ":predict"),
      &model_name, &model_version, &model_version_label, &method,
      &model_subresource, &parse_successful);
  EXPECT_TRUE(parse_successful);
  EXPECT_TRUE(errors::IsInvalidArgument(status));
  EXPECT_THAT(status.error_message(), HasSubstr("Failed to convert version"));
  TF_EXPECT_OK(ParseModelInfo("POST", "/v1/models/foo/metadata", &model_name,
                              &model_version, &model_version_label, &method,
                              &model_subresource, &parse_successful));
  EXPECT_FALSE(parse_successful);
  TF_EXPECT_OK(ParseModelInfo(
      "POST", "/v1/models/foo/versions/50/labels/some_label:regress",
      &model_name, &model_version, &model_version_label, &method,
      &model_subresource, &parse_successful));
  EXPECT_FALSE(parse_successful);
}
}  // namespace
}  // namespace serving
}  // namespace tensorflow
