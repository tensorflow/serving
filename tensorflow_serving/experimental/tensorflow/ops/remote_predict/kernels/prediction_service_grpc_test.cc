/* Copyright 2020 Google Inc. All Rights Reserved.

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
#include "tensorflow_serving/experimental/tensorflow/ops/remote_predict/kernels/prediction_service_grpc.h"

#include <memory>

#include "absl/time/clock.h"
#include "tensorflow/core/framework/tensor_testutil.h"

namespace tensorflow {
namespace serving {
namespace {

class PredictionServiceGrpcTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    auto prediction_service_status =
        PredictionServiceGrpc::Create("target_address", &grpc_stub_);
  }
  std::unique_ptr<PredictionServiceGrpc> grpc_stub_;
  std::unique_ptr<::grpc::ClientContext> rpc_;
};

TEST_F(PredictionServiceGrpcTest, TestSetDeadline) {
  const absl::Duration deadline = absl::Milliseconds(30000);
  auto rpc_or = grpc_stub_->CreateRpc(deadline);
  ASSERT_TRUE(rpc_or.ok());
  rpc_.reset(rpc_or.value());

  EXPECT_NEAR(absl::ToDoubleMilliseconds(deadline),
              absl::ToDoubleMilliseconds(absl::FromChrono(rpc_->deadline()) -
                                         absl::Now()),
              10);
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
