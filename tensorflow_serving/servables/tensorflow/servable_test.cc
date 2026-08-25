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

#include "tensorflow_serving/servables/tensorflow/servable.h"

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorflow_serving/apis/predict.pb.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace {

TEST(EmptyServableTest, Predict) {
  PredictResponse response;
  EXPECT_EQ(EmptyServable()
                .Predict(Servable::RunOptions(), PredictRequest(), &response)
                .code(),
            absl::StatusCode::kFailedPrecondition);
}

TEST(SingleRequestPredictStreamedContextTest, ProcessesOnlyOneRequest) {
  int count = 0;
  SingleRequestPredictStreamedContext context(
      [&count](const PredictRequest& request) -> absl::Status {
        count++;
        return absl::OkStatus();
      });

  PredictRequest request;
  EXPECT_TRUE(context.ProcessRequest(request).ok());
  EXPECT_EQ(count, 1);

  absl::Status status = context.ProcessRequest(request);
  EXPECT_EQ(status.code(), absl::StatusCode::kUnimplemented);
  EXPECT_EQ(count, 1);

  EXPECT_TRUE(context.Close().ok());
}

TEST(SingleRequestPredictStreamedContextTest, CloseWithoutRequestFails) {
  SingleRequestPredictStreamedContext context(
      [](const PredictRequest& request) -> absl::Status {
        return absl::OkStatus();
      });

  absl::Status status = context.Close();
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
}

TEST(HandshakeEnabledPredictStreamedContextTest,
     SuccessWithHandshakeFollowedByPayload) {
  int count = 0;
  HandshakeEnabledPredictStreamedContext context(
      [&count](const PredictRequest& request) -> absl::Status {
        count++;
        return absl::OkStatus();
      });

  PredictRequest handshake_req;
  handshake_req.mutable_request_options()
      ->mutable_handshake()
      ->set_estimated_payload_bytes(100);

  PredictRequest payload_req;

  EXPECT_TRUE(context.ProcessRequest(handshake_req).ok());
  EXPECT_EQ(count, 0);

  EXPECT_TRUE(context.ProcessRequest(payload_req).ok());
  EXPECT_EQ(count, 1);

  EXPECT_TRUE(context.Close().ok());
}

TEST(HandshakeEnabledPredictStreamedContextTest,
     FailureWithHandshakeFollowedByHandshake) {
  HandshakeEnabledPredictStreamedContext context(
      [](const PredictRequest& request) -> absl::Status {
        return absl::OkStatus();
      });

  PredictRequest handshake_req;
  handshake_req.mutable_request_options()
      ->mutable_handshake()
      ->set_estimated_payload_bytes(100);

  EXPECT_TRUE(context.ProcessRequest(handshake_req).ok());

  absl::Status status = context.ProcessRequest(handshake_req);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

TEST(HandshakeEnabledPredictStreamedContextTest,
     FailureWhenFirstIsNotHandshakeAndFollowedBySecond) {
  HandshakeEnabledPredictStreamedContext context(
      [](const PredictRequest& request) -> absl::Status {
        return absl::OkStatus();
      });

  PredictRequest payload_req;

  EXPECT_TRUE(context.ProcessRequest(payload_req).ok());

  absl::Status status = context.ProcessRequest(payload_req);
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
}

TEST(HandshakeEnabledPredictStreamedContextTest, FailureOnRequestThreeOrMore) {
  HandshakeEnabledPredictStreamedContext context(
      [](const PredictRequest& request) -> absl::Status {
        return absl::OkStatus();
      });

  PredictRequest handshake_req;
  handshake_req.mutable_request_options()
      ->mutable_handshake()
      ->set_estimated_payload_bytes(100);

  PredictRequest payload_req;

  EXPECT_TRUE(context.ProcessRequest(handshake_req).ok());
  EXPECT_TRUE(context.ProcessRequest(payload_req).ok());

  absl::Status status = context.ProcessRequest(payload_req);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

TEST(HandshakeEnabledPredictStreamedContextTest, CloseWithoutRequestFails) {
  HandshakeEnabledPredictStreamedContext context(
      [](const PredictRequest& request) -> absl::Status {
        return absl::OkStatus();
      });

  absl::Status status = context.Close();
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
