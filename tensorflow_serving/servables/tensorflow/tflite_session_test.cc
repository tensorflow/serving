/* Copyright 2019 Google Inc. All Rights Reserved.

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

#include "tensorflow_serving/servables/tensorflow/tflite_session.h"

#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace {

constexpr char kTestModel[] =
    "/servables/tensorflow/testdata/saved_model_half_plus_two_tflite/00000123/"
    "model.tflite";

TEST(TfLiteSession, BasicTest) {
  string model_bytes;
  TF_ASSERT_OK(ReadFileToString(tensorflow::Env::Default(),
                                test_util::TestSrcDirPath(kTestModel),
                                &model_bytes));

  ::google::protobuf::Map<string, SignatureDef> signatures;
  std::unique_ptr<TfLiteSession> session;
  TF_EXPECT_OK(
      TfLiteSession::Create(std::move(model_bytes), &session, &signatures));
  EXPECT_EQ(signatures.size(), 1);
  EXPECT_EQ(signatures.begin()->first, "serving_default");
  EXPECT_THAT(signatures.begin()->second, test_util::EqualsProto(R"(
                inputs {
                  key: "x"
                  value {
                    name: "x"
                    dtype: DT_FLOAT
                    tensor_shape {
                      dim { size: 1 }
                      dim { size: 1 }
                    }
                  }
                }
                outputs {
                  key: "y"
                  value {
                    name: "y"
                    dtype: DT_FLOAT
                    tensor_shape {
                      dim { size: 1 }
                      dim { size: 1 }
                    }
                  }
                }
                method_name: "tensorflow/serving/predict"
              )"));
  Tensor input = test::AsTensor<float>({1.0, 2.0, 3.0}, TensorShape({3}));
  {
    // Use TF Lite tensor names.
    std::vector<Tensor> outputs;
    TF_EXPECT_OK(session->Run({{"x", input}}, {"y"}, {}, &outputs));
    ASSERT_EQ(outputs.size(), 1);
    test::ExpectTensorEqual<float>(
        outputs[0], test::AsTensor<float>({2.5, 3, 3.5}, TensorShape({3})));
  }
  {
    // Use TF tensor names (with `:0` suffix).
    std::vector<Tensor> outputs;
    TF_EXPECT_OK(session->Run({{"x:0", input}}, {"y:0"}, {}, &outputs));
    ASSERT_EQ(outputs.size(), 1);
    test::ExpectTensorEqual<float>(
        outputs[0], test::AsTensor<float>({2.5, 3, 3.5}, TensorShape({3})));
  }
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
