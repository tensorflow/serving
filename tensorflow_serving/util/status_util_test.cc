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

#include "tensorflow_serving/util/status_util.h"

#include <gtest/gtest.h>

namespace tensorflow {
namespace serving {
namespace {

TEST(StatusUtilTest, ConvertsErrorStatusToStatusProto) {
  Status status = Status(tensorflow::error::ABORTED, "aborted error message");
  StatusProto status_proto = ToStatusProto(status);
  EXPECT_EQ(tensorflow::error::ABORTED, status_proto.error_code());
  EXPECT_EQ("aborted error message", status_proto.error_message());
}

TEST(StatusUtilTest, ConvertsOkStatusToStatusProto) {
  Status status;
  StatusProto status_proto = ToStatusProto(status);
  EXPECT_EQ(tensorflow::error::OK, status_proto.error_code());
  EXPECT_EQ("", status_proto.error_message());
}

TEST(StatusUtilTest, ConvertsErrorStatusProtoToStatus) {
  StatusProto status_proto;
  status_proto.set_error_code(tensorflow::error::ALREADY_EXISTS);
  status_proto.set_error_message("already exists error message");
  Status status = FromStatusProto(status_proto);
  EXPECT_EQ(tensorflow::error::ALREADY_EXISTS, status.code());
  EXPECT_EQ("already exists error message", status.error_message());
}

TEST(StatusUtilTest, ConvertsOkStatusProtoToStatus) {
  StatusProto status_proto;
  status_proto.set_error_code(tensorflow::error::OK);
  Status status = FromStatusProto(status_proto);
  EXPECT_EQ(tensorflow::error::OK, status.code());
  EXPECT_EQ("", status.error_message());
}

}  // namespace

}  // namespace serving
}  // namespace tensorflow
