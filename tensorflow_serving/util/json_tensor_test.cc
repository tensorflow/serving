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

#include "tensorflow_serving/util/json_tensor.h"

#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"
#include "google/protobuf/util/message_differencer.h"
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/substitute.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow_serving/apis/model.pb.h"
#include "tensorflow_serving/apis/predict.pb.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace {

using protobuf::TextFormat;
using protobuf::util::DefaultFieldComparator;
using protobuf::util::MessageDifferencer;
using test_util::EqualsProto;
using ::testing::HasSubstr;

using TensorInfoMap = ::google::protobuf::Map<string, TensorInfo>;
using TensorMap = ::google::protobuf::Map<string, TensorProto>;

std::function<tensorflow::Status(const string&, TensorInfoMap*)> getmap(
    const TensorInfoMap& map) {
  return [&map](const string&, TensorInfoMap* m) {
    *m = map;
    return Status::OK();
  };
}

TEST(JsontensorTest, SingleUnnamedTensor) {
  TensorInfoMap infomap;
  ASSERT_TRUE(
      TextFormat::ParseFromString("dtype: DT_INT32", &infomap["default"]));

  PredictRequest req;
  TF_EXPECT_OK(FillPredictRequestFromJson(R"(
    {
      "instances": [[1,2],[3,4],[5,6]]
    })",
                                          getmap(infomap), &req));
  auto tmap = req.inputs();
  EXPECT_EQ(tmap.size(), 1);
  EXPECT_THAT(tmap["default"], EqualsProto(R"(
    dtype: DT_INT32
    tensor_shape {
      dim { size: 3 }
      dim { size: 2 }
    }
    int_val: 1
    int_val: 2
    int_val: 3
    int_val: 4
    int_val: 5
    int_val: 6
    )"));
}

TEST(JsontensorTest, SingleUnnamedTensorWithSignature) {
  TensorInfoMap infomap;
  ASSERT_TRUE(
      TextFormat::ParseFromString("dtype: DT_INT32", &infomap["default"]));

  PredictRequest req;
  TF_EXPECT_OK(FillPredictRequestFromJson(R"(
    {
      "signature_name": "predict_images",
      "instances": [[1,2]]
    })",
                                          getmap(infomap), &req));
  EXPECT_EQ(req.model_spec().signature_name(), "predict_images");
  auto tmap = req.inputs();
  EXPECT_EQ(tmap.size(), 1);
  EXPECT_THAT(tmap["default"], EqualsProto(R"(
    dtype: DT_INT32
    tensor_shape {
      dim { size: 1 }
      dim { size: 2 }
    }
    int_val: 1
    int_val: 2
    )"));
}

TEST(JsontensorTest, TensorFromNonNullTerminatedBuffer) {
  TensorInfoMap infomap;
  ASSERT_TRUE(
      TextFormat::ParseFromString("dtype: DT_INT32", &infomap["default"]));

  // Note, last character is a 'X' to for non-null termination.
  const string jsonstr = R"({"instances": [[1,2],[3,4],[5,6]]}X)";
  PredictRequest req;
  TF_EXPECT_OK(FillPredictRequestFromJson(
      // Process over a buffer that is not null terminated.
      absl::string_view(jsonstr.data(), jsonstr.length() - 1), getmap(infomap),
      &req));
  auto tmap = req.inputs();
  EXPECT_EQ(tmap.size(), 1);
  EXPECT_THAT(tmap["default"], EqualsProto(R"(
    dtype: DT_INT32
    tensor_shape {
      dim { size: 3 }
      dim { size: 2 }
    }
    int_val: 1
    int_val: 2
    int_val: 3
    int_val: 4
    int_val: 5
    int_val: 6
    )"));
}

TEST(JsontensorTest, SingleUnnamedTensorBase64Scalars) {
  TensorInfoMap infomap;
  ASSERT_TRUE(
      TextFormat::ParseFromString("dtype: DT_STRING", &infomap["default"]));

  PredictRequest req;
  TF_EXPECT_OK(FillPredictRequestFromJson(R"(
    {
      "instances": [ { "b64" : "aGVsbG8=" }, { "b64": "d29ybGQ=" } ]
    })",
                                          getmap(infomap), &req));
  auto tmap = req.inputs();
  EXPECT_EQ(tmap.size(), 1);
  EXPECT_THAT(tmap["default"], EqualsProto(R"(
    dtype: DT_STRING
    tensor_shape {
      dim { size: 2 }
    }
    string_val: "hello"
    string_val: "world"
    )"));
}

TEST(JsontensorTest, SingleUnnamedTensorBase64Lists) {
  TensorInfoMap infomap;
  ASSERT_TRUE(
      TextFormat::ParseFromString("dtype: DT_STRING", &infomap["default"]));

  PredictRequest req;
  TF_EXPECT_OK(FillPredictRequestFromJson(R"(
    {
      "instances": [ [{ "b64" : "aGVsbG8=" }], [{ "b64": "d29ybGQ=" }] ]
    })",
                                          getmap(infomap), &req));
  auto tmap = req.inputs();
  EXPECT_EQ(tmap.size(), 1);
  EXPECT_THAT(tmap["default"], EqualsProto(R"(
    dtype: DT_STRING
    tensor_shape {
      dim { size: 2 }
      dim { size: 1 }
    }
    string_val: "hello"
    string_val: "world"
    )"));
}

TEST(JsontensorTest, SingleNamedTensorBase64) {
  TensorInfoMap infomap;
  ASSERT_TRUE(
      TextFormat::ParseFromString("dtype: DT_STRING", &infomap["default"]));

  PredictRequest req;
  TF_EXPECT_OK(FillPredictRequestFromJson(R"(
    {
      "instances": [
        {
          "default": [ [{ "b64" : "aGVsbG8=" }], [{ "b64": "d29ybGQ=" }] ]
        }
      ]
    })",
                                          getmap(infomap), &req));
  auto tmap = req.inputs();
  EXPECT_EQ(tmap.size(), 1);
  EXPECT_THAT(tmap["default"], EqualsProto(R"(
    dtype: DT_STRING
    tensor_shape {
      dim { size: 1 }
      dim { size: 2 }
      dim { size: 1 }
    }
    string_val: "hello"
    string_val: "world"
    )"));
}

TEST(JsontensorTest, MultipleNamedTensor) {
  TensorInfoMap infomap;

  // 3 named tensors with different types.
  ASSERT_TRUE(
      TextFormat::ParseFromString("dtype: DT_INT32", &infomap["int_tensor"]));
  ASSERT_TRUE(
      TextFormat::ParseFromString("dtype: DT_STRING", &infomap["str_tensor"]));
  ASSERT_TRUE(
      TextFormat::ParseFromString("dtype: DT_FLOAT", &infomap["float_tensor"]));

  PredictRequest req;
  TF_EXPECT_OK(FillPredictRequestFromJson(R"(
    {
      "instances": [
        {
          "int_tensor": [[1,2],[3,4],[5,6]],
          "str_tensor": ["foo", "bar"],
          "float_tensor": [1.0]
        },
        {
          "int_tensor": [[7,8],[9,0],[1,2]],
          "str_tensor": ["baz", "bat"],
          "float_tensor": [2.0]
        }
      ]
    })",
                                          getmap(infomap), &req));

  auto tmap = req.inputs();
  EXPECT_EQ(tmap.size(), 3);
  EXPECT_THAT(tmap["int_tensor"], EqualsProto(R"(
    dtype: DT_INT32
    tensor_shape {
      dim { size: 2 }
      dim { size: 3 }
      dim { size: 2 }
    }
    int_val: 1
    int_val: 2
    int_val: 3
    int_val: 4
    int_val: 5
    int_val: 6
    int_val: 7
    int_val: 8
    int_val: 9
    int_val: 0
    int_val: 1
    int_val: 2
    )"));
  EXPECT_THAT(tmap["str_tensor"], EqualsProto(R"(
    dtype: DT_STRING
    tensor_shape {
      dim { size: 2 }
      dim { size: 2 }
    }
    string_val: "foo"
    string_val: "bar"
    string_val: "baz"
    string_val: "bat"
    )"));
  EXPECT_THAT(tmap["float_tensor"], EqualsProto(R"(
    dtype: DT_FLOAT
    tensor_shape {
      dim { size: 2 }
      dim { size: 1 }
    }
    float_val: 1.0
    float_val: 2.0
    )"));
}

TEST(JsontensorTest, SingleUnnamedTensorErrors) {
  TensorInfoMap infomap;
  ASSERT_TRUE(
      TextFormat::ParseFromString("dtype: DT_INT32", &infomap["default"]));

  PredictRequest req;
  Status status;
  status = FillPredictRequestFromJson("", getmap(infomap), &req);
  ASSERT_TRUE(errors::IsInvalidArgument(status));
  EXPECT_THAT(status.error_message(), HasSubstr("document is empty"));

  status = FillPredictRequestFromJson(R"(
    {
      "signature_name": 5,
      "instances": [[1,2],[3,4],[5,6,7]]
    })",
                                      getmap(infomap), &req);
  ASSERT_TRUE(errors::IsInvalidArgument(status));
  EXPECT_THAT(status.error_message(), HasSubstr("must be a string value"));

  status = FillPredictRequestFromJson(R"(
    {
      "instances": [[1,2],[3,4],[5,6,7]]
    })",
                                      getmap(infomap), &req);
  ASSERT_TRUE(errors::IsInvalidArgument(status));
  EXPECT_THAT(status.error_message(), HasSubstr("Expecting tensor size"));

  status = FillPredictRequestFromJson(R"(
    {
      "instances": [[1,2],[3,4],[[5,6]]]
    })",
                                      getmap(infomap), &req);
  ASSERT_TRUE(errors::IsInvalidArgument(status));
  EXPECT_THAT(status.error_message(), HasSubstr("Expecting shape"));

  status = FillPredictRequestFromJson(R"(
    {
      "instances": [1, [1]]
    })",
                                      getmap(infomap), &req);
  ASSERT_TRUE(errors::IsInvalidArgument(status));
  EXPECT_THAT(status.error_message(), HasSubstr("Expecting shape"));

  status = FillPredictRequestFromJson(R"(
    {
      "instances": [[1,2],["a", "b"]]
    })",
                                      getmap(infomap), &req);
  ASSERT_TRUE(errors::IsInvalidArgument(status));
  EXPECT_THAT(status.error_message(), HasSubstr("not of expected type"));

  ASSERT_TRUE(
      TextFormat::ParseFromString("dtype: DT_FLOAT", &infomap["default"]));
  status = FillPredictRequestFromJson(
      absl::Substitute(R"({ "instances": [$0] })",
                       std::numeric_limits<double>::max()),
      getmap(infomap), &req);
  ASSERT_TRUE(errors::IsInvalidArgument(status));
  EXPECT_THAT(status.error_message(), HasSubstr("out of range for float"));
}

TEST(JsontensorTest, MultipleNamedTensorErrors) {
  TensorInfoMap infomap;

  // 2 named tensors with different types.
  ASSERT_TRUE(
      TextFormat::ParseFromString("dtype: DT_INT32", &infomap["int_tensor"]));
  ASSERT_TRUE(
      TextFormat::ParseFromString("dtype: DT_STRING", &infomap["str_tensor"]));

  PredictRequest req;
  Status status;
  // Different shapes across int_tensor instances.
  status = FillPredictRequestFromJson(R"(
    {
      "instances": [
        {
          "int_tensor": [[1,2],[3,4],[5,6]],
          "str_tensor": ["foo", "bar"]
        },
        {
          "int_tensor": [[[7,8],[9,0],[1,2]]],
          "str_tensor": ["baz", "bat"]
        }
      ]
    })",
                                      getmap(infomap), &req);
  ASSERT_TRUE(errors::IsInvalidArgument(status));
  EXPECT_THAT(status.error_message(), HasSubstr("Expecting shape"));

  // Different size/length across int_tensor instances.
  req.Clear();
  status = FillPredictRequestFromJson(R"(
    {
      "instances": [
        {
          "int_tensor": [[1,2],[3,4],[5,6],[7,8]],
          "str_tensor": ["foo", "bar"]
        },
        {
          "int_tensor": [[7,8],[9,0],[1,2]],
          "str_tensor": ["baz", "bat"]
        }
      ]
    })",
                                      getmap(infomap), &req);
  ASSERT_TRUE(errors::IsInvalidArgument(status));
  EXPECT_THAT(status.error_message(), HasSubstr("Expecting tensor size"));

  // Mix of object and value/list in "instances" list.
  // First element is an object. Rest are expected to be objects too.
  req.Clear();
  status = FillPredictRequestFromJson(R"(
    {
      "instances": [
        {
          "int_tensor": [[1,2],[3,4],[5,6]],
          "str_tensor": ["foo", "bar"]
        },
        [1, 20, 30],
        {
          "int_tensor": [[[7,8],[9,0],[1,2]]],
          "str_tensor": ["baz", "bat"]
        }
      ]
    })",
                                      getmap(infomap), &req);
  ASSERT_TRUE(errors::IsInvalidArgument(status));
  EXPECT_THAT(status.error_message(),
              HasSubstr("Expecting object but got list"));

  // Mix of object and value/list in "instances" list.
  // First element is a list. Rest are expected to be list too.
  infomap.clear();
  ASSERT_TRUE(
      TextFormat::ParseFromString("dtype: DT_STRING", &infomap["str_tensor"]));
  req.Clear();
  status = FillPredictRequestFromJson(R"(
    {
      "instances": [
        ["baz", "bar"],
        {
          "str_tensor": ["baz", "bat"]
        }
      ]
    })",
                                      getmap(infomap), &req);
  ASSERT_TRUE(errors::IsInvalidArgument(status));
  EXPECT_THAT(status.error_message(),
              HasSubstr("Expecting value/list but got object"));
}

template <const unsigned int parseflags = rapidjson::kParseNanAndInfFlag>
Status CompareJson(const string& json1, const string& json2) {
  rapidjson::Document doc1;
  if (doc1.Parse<parseflags>(json1.c_str()).HasParseError()) {
    return errors::InvalidArgument(
        "LHS JSON Parse error: ",
        rapidjson::GetParseError_En(doc1.GetParseError()),
        " at offset: ", doc1.GetErrorOffset(), " JSON: ", json1);
  }
  rapidjson::Document doc2;
  if (doc2.Parse<parseflags>(json2.c_str()).HasParseError()) {
    return errors::InvalidArgument(
        "RHS JSON Parse error: ",
        rapidjson::GetParseError_En(doc2.GetParseError()),
        " at offset: ", doc2.GetErrorOffset(), " JSON: ", json2);
  }
  if (doc1 != doc2) {
    return errors::InvalidArgument("JSON Different. JSON1: ", json1,
                                   "JSON2: ", json2);
  }
  return Status::OK();
}

// Compare two JSON documents treating values (including numbers) as strings.
//
// Use this if you are comparing JSON with decimal numbers that can have
// NaN or +/-Inf numbers, as comparing them numerically will not work, due
// to the approximate nature of decimal representation.
//
// For most use cases, prefer CompareJson() defined above.
Status CompareJsonAllValuesAsStrings(const string& json1, const string& json2) {
  return CompareJson<rapidjson::kParseNanAndInfFlag |
                     rapidjson::kParseNumbersAsStringsFlag>(json1, json2);
}

TEST(JsontensorTest, FromJsonSingleTensor) {
  TensorMap tensormap;
  ASSERT_TRUE(TextFormat::ParseFromString(R"(
    dtype: DT_INT32
    tensor_shape {
      dim { size: 2 }
      dim { size: 3 }
      dim { size: 2 }
    }
    int_val: 1
    int_val: 2
    int_val: 3
    int_val: 4
    int_val: 5
    int_val: 6
    int_val: 7
    int_val: 8
    int_val: 9
    int_val: 0
    int_val: 1
    int_val: 2
    )",
                                          &tensormap["int_tensor"]));

  string json;
  TF_EXPECT_OK(MakeJsonFromTensors(tensormap, &json));
  TF_EXPECT_OK(CompareJson(json, R"({
    "predictions": [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 0], [1, 2]]
    ]})"));
}

TEST(JsontensorTest, FromJsonSingleScalarTensor) {
  TensorMap tensormap;
  ASSERT_TRUE(TextFormat::ParseFromString(R"(
    dtype: DT_INT32
    tensor_shape {
      dim { size: 5 }
    }
    int_val: 1
    int_val: 2
    int_val: 3
    int_val: 4
    int_val: 5
    )",
                                          &tensormap["int_tensor"]));

  string json;
  TF_EXPECT_OK(MakeJsonFromTensors(tensormap, &json));
  TF_EXPECT_OK(CompareJson(json, R"({ "predictions": [1, 2, 3, 4, 5] })"));
}

TEST(JsontensorTest, FromJsonSingleBytesTensor) {
  TensorMap tensormap;
  ASSERT_TRUE(TextFormat::ParseFromString(R"(
    dtype: DT_STRING
    tensor_shape {
      dim { size: 2 }
      dim { size: 2 }
    }
    string_val: "hello"
    string_val: "world"
    string_val: "tf"
    string_val: "serving"
    )",
                                          &tensormap["str_tensor_bytes"]));

  string json;
  TF_EXPECT_OK(MakeJsonFromTensors(tensormap, &json));
  TF_EXPECT_OK(CompareJson(json, R"({
    "predictions": [
      [{"b64": "aGVsbG8="}, {"b64": "d29ybGQ="}],
      [{"b64": "dGY="}, {"b64": "c2VydmluZw=="}]
    ]})"));
}

TEST(JsontensorTest, FromJsonSingleFloatTensorNonFinite) {
  TensorMap tensormap;
  ASSERT_TRUE(TextFormat::ParseFromString(R"(
    dtype: DT_FLOAT
    tensor_shape {
      dim { size: 2 }
      dim { size: 2 }
    }
    float_val: NaN
    float_val: Infinity
    float_val: 3
    float_val: -Infinity
    )",
                                          &tensormap["float_tensor"]));

  string json;
  TF_EXPECT_OK(MakeJsonFromTensors(tensormap, &json));
  TF_EXPECT_OK(CompareJsonAllValuesAsStrings(json, R"({
    "predictions": [
      [NaN, Infinity],
      [3.0, -Infinity]
    ]})"));
}

TEST(JsontensorTest, FromJsonSingleTensorErrors) {
  TensorMap tensormap;
  string json;
  Status status;

  status = MakeJsonFromTensors(tensormap, &json);
  ASSERT_TRUE(errors::IsInvalidArgument(status));
  EXPECT_THAT(status.error_message(), HasSubstr("empty tensor map"));

  ASSERT_TRUE(TextFormat::ParseFromString(R"(
    dtype: DT_COMPLEX64
    tensor_shape {
      dim { size: 2 }
    }
    scomplex_val: 1.0
    scomplex_val: 2.0
    )",
                                          &tensormap["tensor"]));
  status = MakeJsonFromTensors(tensormap, &json);
  ASSERT_TRUE(errors::IsInvalidArgument(status));
  EXPECT_THAT(status.error_message(), HasSubstr("tensor type: complex64"));

  ASSERT_TRUE(TextFormat::ParseFromString(R"(
    dtype: DT_INT32
    int_val: 1
    )",
                                          &tensormap["tensor"]));
  status = MakeJsonFromTensors(tensormap, &json);
  ASSERT_TRUE(errors::IsInvalidArgument(status));
  EXPECT_THAT(status.error_message(), HasSubstr("no shape information"));
}

TEST(JsontensorTest, FromJsonMultipleNamedTensors) {
  TensorMap tensormap;
  ASSERT_TRUE(TextFormat::ParseFromString(R"(
        dtype: DT_INT32
        tensor_shape {
          dim { size: 2 }
          dim { size: 3 }
          dim { size: 2 }
        }
        int_val: 1
        int_val: 2
        int_val: 3
        int_val: 4
        int_val: 5
        int_val: 6
        int_val: 7
        int_val: 8
        int_val: 9
        int_val: 0
        int_val: 1
        int_val: 2
        )",
                                          &tensormap["int_tensor"]));

  ASSERT_TRUE(TextFormat::ParseFromString(R"(
    dtype: DT_STRING
    tensor_shape {
      dim { size: 2 }
      dim { size: 2 }
    }
    string_val: "foo"
    string_val: "bar"
    string_val: "baz"
    string_val: "bat"
    )",
                                          &tensormap["str_tensor"]));

  ASSERT_TRUE(TextFormat::ParseFromString(R"(
    dtype: DT_FLOAT
    tensor_shape {
      dim { size: 2 }
      dim { size: 1 }
    }
    float_val: 1.0
    float_val: 2.0
    )",
                                          &tensormap["float_tensor"]));

  ASSERT_TRUE(TextFormat::ParseFromString(R"(
    dtype: DT_DOUBLE
    tensor_shape {
      dim { size: 2 }
    }
    double_val: 8.0
    double_val: 9.0
    )",
                                          &tensormap["double_scalar_tensor"]));

  ASSERT_TRUE(TextFormat::ParseFromString(R"(
    dtype: DT_STRING
    tensor_shape {
      dim { size: 2 }
      dim { size: 2 }
    }
    string_val: "hello"
    string_val: "world"
    string_val: "tf"
    string_val: "serving"
    )",
                                          &tensormap["str_tensor_bytes"]));

  string json;
  TF_EXPECT_OK(MakeJsonFromTensors(tensormap, &json));
  TF_EXPECT_OK(CompareJson(json, R"({
    "predictions": [
        {
            "double_scalar_tensor": 8.0,
            "float_tensor": [1.0],
            "int_tensor": [[1, 2], [3, 4], [5, 6]],
            "str_tensor": ["foo", "bar"],
            "str_tensor_bytes": [ {"b64": "aGVsbG8="}, {"b64": "d29ybGQ="} ]
        },
        {
            "double_scalar_tensor": 9.0,
            "float_tensor": [2.0],
            "int_tensor": [[7, 8], [9, 0], [1, 2]],
            "str_tensor": ["baz", "bat"],
            "str_tensor_bytes": [ {"b64": "dGY="}, {"b64": "c2VydmluZw=="} ]
        }
    ]})"));
}

TEST(JsontensorTest, FromJsonMultipleNamedTensorsErrors) {
  TensorMap tensormap;
  ASSERT_TRUE(TextFormat::ParseFromString(R"(
    dtype: DT_INT32
    tensor_shape {
      dim { size: 2 }
      dim { size: 3 }
    }
    int_val: 1
    int_val: 2
    int_val: 3
    int_val: 4
    int_val: 5
    int_val: 6
    )",
                                          &tensormap["int_tensor"]));

  ASSERT_TRUE(TextFormat::ParseFromString(R"(
    dtype: DT_STRING
    tensor_shape {
      dim { size: 1 }
      dim { size: 3 }
    }
    string_val: "foo"
    string_val: "bar"
    string_val: "baz"
    )",
                                          &tensormap["str_tensor"]));

  string json;
  const auto& status = MakeJsonFromTensors(tensormap, &json);
  ASSERT_TRUE(errors::IsInvalidArgument(status));
  EXPECT_THAT(status.error_message(), HasSubstr("inconsistent batch size"));
}

template <typename RequestType>
class ClassifyRegressRequestTest : public ::testing::Test {
 protected:
  Status FillRequest(const string& json, ClassificationRequest* req) {
    return FillClassificationRequestFromJson(json, req);
  }

  Status FillRequest(const string& json, RegressionRequest* req) {
    return FillRegressionRequestFromJson(json, req);
  }
};

typedef ::testing::Types<ClassificationRequest, RegressionRequest> RequestTypes;
TYPED_TEST_CASE(ClassifyRegressRequestTest, RequestTypes);

TYPED_TEST(ClassifyRegressRequestTest, RequestNoContext) {
  TypeParam req;
  TF_EXPECT_OK(this->FillRequest(R"(
    {
      "examples": [
        {
          "names": [ "foo", "bar" ],
          "ratings": [ 8.0, 9.0 ],
          "age": 20
        },
        {
          "names": [ "baz" ],
          "ratings": 6.0
        }
      ]
    })",
                                 &req));

  TypeParam expected_req;
  ASSERT_TRUE(TextFormat::ParseFromString(R"(
    input {
      example_list {
        examples {
          features {
            feature {
              key: "names"
              value { bytes_list {
                value: "foo"
                value: "bar"
              }}
            }
            feature {
              key: "ratings"
              value { float_list {
                value: 8.0
                value: 9.0
              }}
            }
            feature {
              key: "age"
              value { int64_list {
                value: 20
              }}
            }
          }
        }
        examples {
          features {
            feature {
              key: "names"
              value { bytes_list {
                value: "baz"
              }}
            }
            feature {
              key: "ratings"
              value { float_list {
                value: 6.0
              }}
            }
          }
        }
      }
    }
  )",
                                          &expected_req));
  EXPECT_TRUE(MessageDifferencer::ApproximatelyEquals(req, expected_req))
      << "Expected Proto: " << expected_req.DebugString();
}

TYPED_TEST(ClassifyRegressRequestTest, RequestWithContextAndSignature) {
  TypeParam req;
  TF_EXPECT_OK(this->FillRequest(R"(
    {
      "signature_name": "custom_signture",
      "context": {
        "query": "pizza",
        "location": [ "sfo" ]
      },
      "examples": [
        {
          "names": [ "foo", "bar" ],
          "ratings": [ 8.0, 9.0, NaN, Infinity ],
          "age": 20
        },
        {
          "names": [ "baz", { "b64": "aGVsbG8=" } ],
          "ratings": 6.0,
          "intboolmix": [ 1, true, false ]
        },
        {
          "names": "bar",
          "ratings": -Infinity
        }
      ]
    })",
                                 &req));

  TypeParam expected_req;
  ASSERT_TRUE(TextFormat::ParseFromString(R"(
    model_spec {
      signature_name: "custom_signture"
    }
    input {
      example_list_with_context {
        context {
          features {
            feature {
              key: "query"
              value { bytes_list {
                value: "pizza"
              }}
            }
            feature {
              key: "location"
              value { bytes_list {
                value: "sfo"
              }}
            }
          }
        }
        examples {
          features {
            feature {
              key: "names"
              value { bytes_list {
                value: "foo"
                value: "bar"
              }}
            }
            feature {
              key: "ratings"
              value { float_list {
                value: 8.0
                value: 9.0
                value: NaN
                value: Infinity
              }}
            }
            feature {
              key: "age"
              value { int64_list {
                value: 20
              }}
            }
          }
        }
        examples {
          features {
            feature {
              key: "names"
              value { bytes_list {
                value: "baz"
                value: "hello"
              }}
            }
            feature {
              key: "ratings"
              value { float_list {
                value: 6.0
              }}
            }
            feature {
              key: "intboolmix"
              value { int64_list {
                value: 1
                value: 1
                value: 0
              }}
            }
          }
        }
        examples {
          features {
            feature {
              key: "names"
              value { bytes_list {
                value: "bar"
              }}
            }
            feature {
              key: "ratings"
              value { float_list {
                value: -Infinity
              }}
            }
          }
        }

      }
    }
  )",
                                          &expected_req));
  DefaultFieldComparator comparator;
  comparator.set_float_comparison(DefaultFieldComparator::APPROXIMATE);
  comparator.set_treat_nan_as_equal(true);
  MessageDifferencer differencer;
  differencer.set_field_comparator(&comparator);
  EXPECT_TRUE(differencer.Compare(req, expected_req))
      << "Expected proto: " << expected_req.DebugString()
      << "But got proto: " << req.DebugString();
}

TYPED_TEST(ClassifyRegressRequestTest, JsonErrors) {
  TypeParam req;
  auto status = this->FillRequest(R"(
    {
      "signature_name": [ "hello" ],
      "examples": [ { "names": [ "foo", "bar" ] } ]
    })",
                                  &req);
  ASSERT_TRUE(errors::IsInvalidArgument(status));
  EXPECT_THAT(status.error_message(),
              HasSubstr("'signature_name' key must be a string"));

  req.Clear();
  status = this->FillRequest(R"(
    {
      "context": [ { "names": [ "foo", "bar" ] } ]
    })",
                             &req);
  ASSERT_TRUE(errors::IsInvalidArgument(status));
  EXPECT_THAT(status.error_message(), HasSubstr("Example must be JSON object"));

  req.Clear();
  status = this->FillRequest(R"(
    {
      "examples": { "names": [ "foo", "bar" ] }
    })",
                             &req);
  ASSERT_TRUE(errors::IsInvalidArgument(status));
  EXPECT_THAT(status.error_message(), HasSubstr("list/array as the value"));

  req.Clear();
  status = this->FillRequest(R"(
    {
      "examples": [ [ { "names": [ "foo", "bar" ] } ] ]
    })",
                             &req);
  ASSERT_TRUE(errors::IsInvalidArgument(status));
  EXPECT_THAT(status.error_message(), HasSubstr("Example must be JSON object"));

  req.Clear();
  status = this->FillRequest(R"(
    {
      "examples": [ { "names": [ 10, null ] } ]
    })",
                             &req);
  ASSERT_TRUE(errors::IsInvalidArgument(status));
  EXPECT_THAT(status.error_message(),
              HasSubstr("names has element with unexpected JSON type: Null"));

  req.Clear();
  status = this->FillRequest(R"(
    {
      "examples": [ { "names": [ 10, 10.0 ] } ]
    })",
                             &req);
  ASSERT_TRUE(errors::IsInvalidArgument(status));
  EXPECT_THAT(status.error_message(),
              HasSubstr("feature: names expecting type: int64"));

  req.Clear();
  status = this->FillRequest(R"(
    {
      "examples": [ { "names": [ 10, { "test": 10 } ] } ]
    })",
                             &req);
  ASSERT_TRUE(errors::IsInvalidArgument(status));
  EXPECT_THAT(status.error_message(),
              HasSubstr("names has element with unexpected JSON type: Object"));

  req.Clear();
  status = this->FillRequest(R"(
    {
      "examples": [ { "names": [ [10], 20 ] } ]
    })",
                             &req);
  ASSERT_TRUE(errors::IsInvalidArgument(status));
  EXPECT_THAT(status.error_message(),
              HasSubstr("names has element with unexpected JSON type: Array"));

  req.Clear();
  status = this->FillRequest(R"(
    {
      "examples": [ { "names": [ 20, 18446744073709551603 ] } ]
    })",
                             &req);
  ASSERT_TRUE(errors::IsInvalidArgument(status));
  EXPECT_THAT(status.error_message(), HasSubstr("Only int64 is supported"));

  req.Clear();
  status = this->FillRequest(
      absl::Substitute(R"({ "examples": [ { "names": $0 } ] })",
                       std::numeric_limits<double>::max()),
      &req);
  ASSERT_TRUE(errors::IsInvalidArgument(status));
  EXPECT_THAT(status.error_message(), HasSubstr("out of range for float"));
}

TEST(ClassifyRegressnResultTest, JsonFromClassificationResult) {
  ClassificationResult result;
  ASSERT_TRUE(TextFormat::ParseFromString(R"(
    classifications {
      classes { label: "car" score: 0.2 }
      classes { label: "bike" score: 0.7 }
      classes { label: "bus" score: 0.1 }
    }
    classifications {
      classes { label: "car" score: 0.7 }
      classes { label: "bike" score: 0.1 }
      classes { label: "bus" score: 0.2 }
    })",
                                          &result));

  string json;
  TF_EXPECT_OK(MakeJsonFromClassificationResult(result, &json));
  TF_EXPECT_OK(CompareJson(json, R"({
    "results": [
      [ ["car", 0.2], ["bike", 0.7], ["bus", 0.1] ],
      [ ["car", 0.7], ["bike", 0.1], ["bus", 0.2] ]
    ]
  })"));
}

TEST(ClassifyRegressnResultTest, JsonFromRegressionResult) {
  RegressionResult result;
  ASSERT_TRUE(TextFormat::ParseFromString(R"(
    regressions { value: 0.2 }
    regressions { value: 0.9 }
    regressions { value: 1.0 }
    )",
                                          &result));

  string json;
  TF_EXPECT_OK(MakeJsonFromRegressionResult(result, &json));
  TF_EXPECT_OK(CompareJson(json, R"({ "results": [ 0.2, 0.9, 1.0 ] })"));
}

TEST(ClassifyRegressnResultTest, JsonFromRegressionResultWithNonFinite) {
  RegressionResult result;
  ASSERT_TRUE(TextFormat::ParseFromString(R"(
    regressions { value: 0.2 }
    regressions { value: Infinity }
    regressions { value: 1.0 }
    regressions { value: -Infinity }
    regressions { value: NaN }
    )",
                                          &result));

  string json;
  TF_EXPECT_OK(MakeJsonFromRegressionResult(result, &json));
  TF_EXPECT_OK(CompareJsonAllValuesAsStrings(
      json, R"({ "results": [ 0.2, Infinity, 1.0, -Infinity, NaN ] })"));
}

TEST(ClassifyRegressnResultTest, JsonFromResultErrors) {
  string json;
  auto status = MakeJsonFromClassificationResult(ClassificationResult(), &json);
  ASSERT_TRUE(errors::IsInvalidArgument(status));
  EXPECT_THAT(status.error_message(), HasSubstr("empty ClassificationResults"));

  status = MakeJsonFromRegressionResult(RegressionResult(), &json);
  ASSERT_TRUE(errors::IsInvalidArgument(status));
  EXPECT_THAT(status.error_message(), HasSubstr("empty RegressionResults"));
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
