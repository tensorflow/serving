/* Copyright 2022 Google Inc. All Rights Reserved.

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
#include "tensorflow_serving/util/proto_util.h"

#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow_serving/apis/status.pb.h"

namespace tensorflow {
namespace serving {
namespace {

TEST(ProtoUtilTest, ParseProtoTextFile_Errors) {
  StatusProto status_proto;
  auto status = ParseProtoTextFile<StatusProto>("missing.txt", &status_proto);
  ASSERT_TRUE(errors::IsNotFound(status));

  const string kProtoFile = io::JoinPath(testing::TmpDir(), "corrupt.txt");
  std::unique_ptr<WritableFile> file;
  TF_ASSERT_OK(Env::Default()->NewWritableFile(kProtoFile, &file));
  TF_ASSERT_OK(file->Append("corrupt proto content"));
  TF_ASSERT_OK(file->Close());
  status = ParseProtoTextFile<StatusProto>(kProtoFile, &status_proto);
  ASSERT_TRUE(errors::IsInvalidArgument(status));
}

TEST(ProtoUtilTest, ParseProtoTextFile_EmptyFile_Success) {
  StatusProto status_proto;
  const string kProtoFile = io::JoinPath(testing::TmpDir(), "proto1.txt");
  std::unique_ptr<WritableFile> file;
  TF_ASSERT_OK(Env::Default()->NewWritableFile(kProtoFile, &file));
  TF_ASSERT_OK(file->Append(" "));
  TF_ASSERT_OK(file->Close());
  auto status = ParseProtoTextFile<StatusProto>(kProtoFile, &status_proto);
  TF_ASSERT_OK(status);
}

TEST(ProtoUtilTest, ParseProtoTextFile_Success) {
  StatusProto status_proto;
  const string kProtoFile = io::JoinPath(testing::TmpDir(), "proto2.txt");
  std::unique_ptr<WritableFile> file;
  TF_ASSERT_OK(Env::Default()->NewWritableFile(kProtoFile, &file));
  TF_ASSERT_OK(file->Append("error_message: 'hello'"));
  TF_ASSERT_OK(file->Close());
  auto status = ParseProtoTextFile<StatusProto>(kProtoFile, &status_proto);
  TF_ASSERT_OK(status);
  ASSERT_EQ("hello", status_proto.error_message());
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
