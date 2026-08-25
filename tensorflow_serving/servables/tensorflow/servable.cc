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

#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tensorflow_serving/apis/predict.pb.h"

namespace tensorflow {
namespace serving {

bool Servable::SupportsPaging() const { return false; }

absl::Status Servable::Suspend() {
  return absl::UnimplementedError("paging not supported");
}

absl::Status Servable::Resume() {
  return absl::UnimplementedError("paging not supported");
}

EmptyServable::EmptyServable()
    : Servable(/*name=*/"", /*version=*/0),
      error_(absl::FailedPreconditionError("No models loaded")) {}

SingleRequestPredictStreamedContext::SingleRequestPredictStreamedContext(
    absl::AnyInvocable<absl::Status(const PredictRequest&)> f)
    : f_(std::move(f)) {}

absl::Status SingleRequestPredictStreamedContext::ProcessRequest(
    const PredictRequest& request) {
  if (request.has_request_options() &&
      request.request_options().has_handshake()) {
    return absl::InvalidArgumentError(
        "Handshaking is not supported in this context.");
  }
  if (one_request_received_) {
    return absl::UnimplementedError(
        "PredictStreamed already received one request. Accepting more than "
        "one request in a stream is not supported yet");
  }
  one_request_received_ = true;
  return f_(request);
}

absl::Status SingleRequestPredictStreamedContext::Close() {
  if (!one_request_received_) {
    return absl::FailedPreconditionError(
        "PredictStreamed requires at least one request");
  }
  return absl::OkStatus();
}

absl::Status SingleRequestPredictStreamedContext::WaitResponses() {
  return absl::OkStatus();
}

HandshakeEnabledPredictStreamedContext::HandshakeEnabledPredictStreamedContext(
    absl::AnyInvocable<absl::Status(const PredictRequest&)> f)
    : f_(std::move(f)) {}

absl::Status HandshakeEnabledPredictStreamedContext::ProcessRequest(
    const PredictRequest& request) {
  request_count_++;
  if (request_count_ == 1) {
    first_is_handshake_ = request.has_request_options() &&
                          request.request_options().has_handshake();
    if (first_is_handshake_) {
      return absl::OkStatus();
    }
    return f_(request);
  } else if (request_count_ == 2) {
    if (!first_is_handshake_) {
      return absl::FailedPreconditionError(
          "HandshakeEnabledPredictStreamed accepts multiple requests (2) only "
          "if a handshake was received as the first request.");
    }
    if (request.has_request_options() &&
        request.request_options().has_handshake()) {
      return absl::InvalidArgumentError(
          "Followup request after a handshake request must not have the "
          "handshake field set.");
    }
    return f_(request);
  }
  return absl::InvalidArgumentError(
      "PredictStreamed session allows at most 2 requests when handshake is "
      "used.");
}

absl::Status HandshakeEnabledPredictStreamedContext::Close() {
  if (request_count_ == 0) {
    return absl::FailedPreconditionError(
        "PredictStreamed requires at least one request");
  }
  return absl::OkStatus();
}

absl::Status HandshakeEnabledPredictStreamedContext::WaitResponses() {
  return absl::OkStatus();
}

}  // namespace serving
}  // namespace tensorflow
