/* Copyright 2018 Google Inc. All Rights Reserved.

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

#include "tensorflow_serving/model_servers/http_rest_prediction_handler.h"

#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "re2/re2.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow_serving/core/availability_preserving_policy.h"
#include "tensorflow_serving/model_servers/model_platform_types.h"
#include "tensorflow_serving/model_servers/platform_config_util.h"
#include "tensorflow_serving/model_servers/server_core.h"
#include "tensorflow_serving/servables/tensorflow/saved_model_bundle_source_adapter.pb.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_config.pb.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_source_adapter.pb.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace {

using ::testing::HasSubstr;
using ::testing::Not;
using ::testing::UnorderedElementsAreArray;

constexpr char kTestModelBasePath[] = "cc/saved_model/testdata/half_plus_two";
constexpr char kTestModelName[] = "saved_model_half_plus_two_2_versions";
constexpr char kNonexistentModelName[] = "nonexistent_model";
constexpr int kTestModelVersion1 = 123;

using HeaderList = std::vector<std::pair<string, string>>;

class HttpRestPredictionHandlerTest : public ::testing::Test {
 public:
  static void SetUpTestCase() {
    TF_ASSERT_OK(CreateServerCore(&server_core_));

    const int total = 1;  // Number of models expected to be loaded.
    int count = 0;
    while ((count = server_core_->ListAvailableServableIds().size()) < total) {
      LOG(INFO) << "Available servables: " << count << " waiting for " << total;
      absl::SleepFor(absl::Milliseconds(500));
    }
    for (const auto& s : server_core_->ListAvailableServableIds()) {
      LOG(INFO) << "Available servable: " << s.DebugString();
    }
  }

  static void TearDownTestCase() { server_core_.reset(); }

 protected:
  HttpRestPredictionHandlerTest() : handler_(RunOptions(), GetServerCore()) {}

  static Status CreateServerCore(std::unique_ptr<ServerCore>* server_core) {
    ModelServerConfig config;
    auto* model_config = config.mutable_model_config_list()->add_config();
    model_config->set_name(kTestModelName);
    model_config->set_base_path(
        test_util::TensorflowTestSrcDirPath(kTestModelBasePath));
    auto* specific_versions =
        model_config->mutable_model_version_policy()->mutable_specific();
    specific_versions->add_versions(kTestModelVersion1);

    model_config->set_model_platform(kTensorFlowModelPlatform);

    // For ServerCore Options, we leave servable_state_monitor_creator
    // unspecified so the default servable_state_monitor_creator will be used.
    ServerCore::Options options;
    options.model_server_config = config;

    options.platform_config_map = CreateTensorFlowPlatformConfigMap(
        SessionBundleConfig(), true /* use_saved_model */);
    // Reduce the number of initial load threads to be num_load_threads to avoid
    // timing out in tests.
    options.num_initial_load_threads = options.num_load_threads;
    options.aspired_version_policy =
        std::unique_ptr<AspiredVersionPolicy>(new AvailabilityPreservingPolicy);
    return ServerCore::Create(std::move(options), server_core);
  }

  string GetJsonErrorMsg(const string& json) {
    rapidjson::Document doc;
    if (doc.Parse(json.c_str()).HasParseError()) {
      return absl::StrCat("JSON Parse error: ",
                          rapidjson::GetParseError_En(doc.GetParseError()),
                          " for doc: ", json);
    }
    if (!doc.IsObject()) {
      return absl::StrCat("JSON does not have top-level object: ", json);
    }
    const auto itr = doc.FindMember("error");
    if (itr == doc.MemberEnd() || !itr->value.IsString()) {
      return absl::StrCat("JSON object does not have \'error\' key: ", json);
    }
    string escaped_errmsg;
    escaped_errmsg.assign(itr->value.GetString(), itr->value.GetStringLength());
    string errmsg;
    string unescaping_error;
    if (!absl::CUnescape(escaped_errmsg, &errmsg, &unescaping_error)) {
      return absl::StrCat("Error unescaping JSON error message: ",
                          unescaping_error);
    }
    return errmsg;
  }

  ServerCore* GetServerCore() { return server_core_.get(); }

  HttpRestPredictionHandler handler_;

 private:
  static std::unique_ptr<ServerCore> server_core_;
};

std::unique_ptr<ServerCore> HttpRestPredictionHandlerTest::server_core_;

Status CompareJson(const string& json1, const string& json2) {
  rapidjson::Document doc1;
  if (doc1.Parse(json1.c_str()).HasParseError()) {
    return errors::InvalidArgument(
        "JSON Parse error: ", rapidjson::GetParseError_En(doc1.GetParseError()),
        " at offset: ", doc1.GetErrorOffset(), " JSON: ", json1);
  }
  rapidjson::Document doc2;
  if (doc2.Parse(json2.c_str()).HasParseError()) {
    return errors::InvalidArgument(
        "JSON Parse error: ", rapidjson::GetParseError_En(doc2.GetParseError()),
        " at offset: ", doc2.GetErrorOffset(), " JSON: ", json2);
  }
  if (doc1 != doc2) {
    return errors::InvalidArgument("JSON Different. JSON1: ", json1,
                                   "JSON2: ", json2);
  }
  return Status::OK();
}

TEST_F(HttpRestPredictionHandlerTest, kPathRegex) {
  EXPECT_TRUE(RE2::FullMatch("/v1/models", handler_.kPathRegex));
  EXPECT_FALSE(RE2::FullMatch("/statuspage", handler_.kPathRegex));
  EXPECT_FALSE(RE2::FullMatch("/index", handler_.kPathRegex));
}

TEST_F(HttpRestPredictionHandlerTest, UnsupportedApiCalls) {
  HeaderList headers;
  string output;
  Status status;
  status = handler_.ProcessRequest("GET", "/v1/foo", "", &headers, &output);
  EXPECT_TRUE(errors::IsInvalidArgument(status));
  EXPECT_THAT(status.error_message(), HasSubstr("Malformed request"));

  status = handler_.ProcessRequest("POST", "/v1/foo", "", &headers, &output);
  EXPECT_TRUE(errors::IsInvalidArgument(status));
  EXPECT_THAT(status.error_message(), HasSubstr("Malformed request"));

  status = handler_.ProcessRequest("GET", "/v1/models", "", &headers, &output);
  EXPECT_TRUE(errors::IsInvalidArgument(status));
  EXPECT_THAT(status.error_message(), HasSubstr("Malformed request"));

  status = handler_.ProcessRequest("POST", "/v1/models", "", &headers, &output);
  EXPECT_TRUE(errors::IsInvalidArgument(status));
  EXPECT_THAT(status.error_message(), HasSubstr("Malformed request"));

  status =
      handler_.ProcessRequest("GET", "/v1/models/foo", "", &headers, &output);
  EXPECT_TRUE(errors::IsInvalidArgument(status));
  EXPECT_THAT(status.error_message(), HasSubstr("Malformed request"));

  status = handler_.ProcessRequest("GET", "/v1/models/foo/version/50", "",
                                   &headers, &output);
  EXPECT_TRUE(errors::IsInvalidArgument(status));
  EXPECT_THAT(status.error_message(), HasSubstr("Malformed request"));

  status = handler_.ProcessRequest("GET", "/v1/models/foo:predict", "",
                                   &headers, &output);
  EXPECT_TRUE(errors::IsInvalidArgument(status));
  EXPECT_THAT(status.error_message(), HasSubstr("Malformed request"));

  status = handler_.ProcessRequest("GET", "/v1/models/foo/version/50:predict",
                                   "", &headers, &output);
  EXPECT_TRUE(errors::IsInvalidArgument(status));
  EXPECT_THAT(status.error_message(), HasSubstr("Malformed request"));

  status = handler_.ProcessRequest("POST", "/v1/models/foo/version/50:regress",
                                   "", &headers, &output);
  EXPECT_TRUE(errors::IsInvalidArgument(status));
  EXPECT_THAT(status.error_message(), HasSubstr("Malformed request"));

  status = handler_.ProcessRequest(
      "POST", "/v1/models/foo/versions/HELLO:regress", "", &headers, &output);
  EXPECT_TRUE(errors::IsInvalidArgument(status));
  EXPECT_THAT(status.error_message(), HasSubstr("Malformed request"));

  status = handler_.ProcessRequest(
      "POST",
      absl::StrCat("/v1/models/foo/versions/",
                   std::numeric_limits<uint64>::max(), ":regress"),
      "", &headers, &output);
  EXPECT_TRUE(errors::IsInvalidArgument(status));
  EXPECT_THAT(status.error_message(), HasSubstr("Failed to convert version"));
}

TEST_F(HttpRestPredictionHandlerTest, PredictModelNameVersionErrors) {
  HeaderList headers;
  string output;
  Status status;
  // 'foo' is not a valid model name.
  status =
      handler_.ProcessRequest("POST", "/v1/models/foo:predict",
                              R"({ "instances": [1] })", &headers, &output);
  EXPECT_TRUE(errors::IsNotFound(status));

  // 'foo' is not a valid model name.
  status =
      handler_.ProcessRequest("POST", "/v1/models/foo/versions/50:predict",
                              R"({ "instances": [1] })", &headers, &output);
  EXPECT_TRUE(errors::IsNotFound(status));

  // Valid model name, but invalid version number (99).
  status = handler_.ProcessRequest(
      "POST", absl::StrCat("/v1/models/", kTestModelName, "99:predict"),
      R"({ "instances": [1] })", &headers, &output);
  EXPECT_TRUE(errors::IsNotFound(status));
}

TEST_F(HttpRestPredictionHandlerTest, PredictRequestErrors) {
  HeaderList headers;
  string output;
  Status status;
  const string& req_path =
      absl::StrCat("/v1/models/", kTestModelName, ":predict");

  // Empty document.
  status = handler_.ProcessRequest("POST", req_path, "", &headers, &output);
  EXPECT_TRUE(errors::IsInvalidArgument(status));
  EXPECT_THAT(status.error_message(),
              HasSubstr("JSON Parse error: The document is empty"));

  // Badly formatted JSON.
  status = handler_.ProcessRequest("POST", req_path, "instances = [1, 2]",
                                   &headers, &output);
  EXPECT_TRUE(errors::IsInvalidArgument(status));
  EXPECT_THAT(status.error_message(),
              HasSubstr("JSON Parse error: Invalid value"));

  // Incorrect type.
  status = handler_.ProcessRequest(
      "POST", req_path, R"({ "instances": [1, 2] })", &headers, &output);
  EXPECT_TRUE(errors::IsInvalidArgument(status));
  EXPECT_THAT(status.error_message(), HasSubstr("not of expected type: float"));
}

TEST_F(HttpRestPredictionHandlerTest, Predict) {
  HeaderList headers;
  string output;
  Status status;
  // Query latest version.
  TF_EXPECT_OK(handler_.ProcessRequest(
      "POST", absl::StrCat("/v1/models/", kTestModelName, ":predict"),
      R"({"instances": [[1.0, 2.0], [3.0, 4.0]]})", &headers, &output));
  TF_EXPECT_OK(
      CompareJson(output, R"({ "predictions": [[2.5, 3.0], [3.5, 4.0]] })"));
  EXPECT_THAT(headers, UnorderedElementsAreArray(
                           (HeaderList){{"Content-Type", "application/json"}}));

  // Query specific versions.
  TF_EXPECT_OK(handler_.ProcessRequest(
      "POST",
      absl::StrCat("/v1/models/", kTestModelName, "/versions/",
                   kTestModelVersion1, ":predict"),
      R"({"instances": [1.0, 2.0]})", &headers, &output));
  TF_EXPECT_OK(CompareJson(output, R"({ "predictions": [2.5, 3.0] })"));
  EXPECT_THAT(headers, UnorderedElementsAreArray(
                           (HeaderList){{"Content-Type", "application/json"}}));

  // Query specific versions with explicit signature_name.
  TF_EXPECT_OK(handler_.ProcessRequest(
      "POST",
      absl::StrCat("/v1/models/", kTestModelName, "/versions/",
                   kTestModelVersion1, ":predict"),
      R"({
          "signature_name": "serving_default",
          "instances": [3.0, 4.0]})",
      &headers, &output));
  TF_EXPECT_OK(CompareJson(output, R"({ "predictions": [3.5, 4.0] })"));
  EXPECT_THAT(headers, UnorderedElementsAreArray(
                           (HeaderList){{"Content-Type", "application/json"}}));
}

TEST_F(HttpRestPredictionHandlerTest, Regress) {
  HeaderList headers;
  string output;
  Status status;
  // Query latest version.
  TF_EXPECT_OK(handler_.ProcessRequest(
      "POST", absl::StrCat("/v1/models/", kTestModelName, ":regress"),
      R"({
        "signature_name": "regress_x_to_y",
        "examples": [ { "x": 80.0 } ]
        })",
      &headers, &output));
  TF_EXPECT_OK(CompareJson(output, R"({ "results": [42] })"));
  EXPECT_THAT(headers, UnorderedElementsAreArray(
                           (HeaderList){{"Content-Type", "application/json"}}));

  // Query specific versions.
  TF_EXPECT_OK(handler_.ProcessRequest(
      "POST",
      absl::StrCat("/v1/models/", kTestModelName, "/versions/",
                   kTestModelVersion1, ":regress"),
      R"({
        "signature_name": "regress_x_to_y",
        "examples": [ { "x": 80.0 } ]
        })",
      &headers, &output));
  TF_EXPECT_OK(CompareJson(output, R"({ "results": [42] })"));
  EXPECT_THAT(headers, UnorderedElementsAreArray(
                           (HeaderList){{"Content-Type", "application/json"}}));
}

TEST_F(HttpRestPredictionHandlerTest, Classify) {
  HeaderList headers;
  string output;
  Status status;
  // Query latest version.
  TF_EXPECT_OK(handler_.ProcessRequest(
      "POST", absl::StrCat("/v1/models/", kTestModelName, ":classify"),
      R"({
        "signature_name": "classify_x_to_y",
        "examples": [ { "x": 20.0 } ]
        })",
      &headers, &output));
  TF_EXPECT_OK(CompareJson(output, R"({ "results": [[["", 12]]] })"));
  EXPECT_THAT(headers, UnorderedElementsAreArray(
                           (HeaderList){{"Content-Type", "application/json"}}));

  // Query specific versions.
  TF_EXPECT_OK(handler_.ProcessRequest(
      "POST",
      absl::StrCat("/v1/models/", kTestModelName, "/versions/",
                   kTestModelVersion1, ":classify"),
      R"({
        "signature_name": "classify_x_to_y",
        "examples": [ { "x": 10.0 } ]
        })",
      &headers, &output));
  TF_EXPECT_OK(CompareJson(output, R"({ "results": [[["", 7]]] })"));
  EXPECT_THAT(headers, UnorderedElementsAreArray(
                           (HeaderList){{"Content-Type", "application/json"}}));
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
