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

#include "tensorflow_serving/session_bundle/session_bundle_util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace session_bundle {
namespace {

using ::testing::HasSubstr;

const char kTestSavedModelPath[] =
    "cc/saved_model/testdata/half_plus_two/00000123";

const char kTestSessionBundleExportPath[] =
    "session_bundle/testdata/half_plus_two/00000123";

TEST(SessionBundleTest, ConvertSignaturesToSignatureDefsTest) {
  MetaGraphDef meta_graph_def;
  Status status = ConvertSignaturesToSignatureDefs(&meta_graph_def);
  EXPECT_EQ(::tensorflow::error::UNIMPLEMENTED, status.code());
  EXPECT_THAT(
      status.ToString(),
      ::testing::HasSubstr("Session Bundle is deprecated and removed."));
}

TEST(SessionBundleTest, ConvertSessionBundleToSavedModelBundleTest) {
  SessionBundle session_bundle;
  SavedModelBundle saved_model_bundle;
  Status status = session_bundle::ConvertSessionBundleToSavedModelBundle(
      session_bundle, &saved_model_bundle);
  EXPECT_EQ(::tensorflow::error::UNIMPLEMENTED, status.code());
  EXPECT_THAT(
      status.ToString(),
      ::testing::HasSubstr("Session Bundle is deprecated and removed."));
}

TEST(SessionBundleTest, LoadSessionBundleOrSavedModelBundleTest) {
  SessionOptions session_options;
  RunOptions run_options;
  SavedModelBundle bundle;
  bool is_session_bundle;
  const std::unordered_set<string> tags = {"serve"};
  const string export_dir =
      test_util::TensorflowTestSrcDirPath(kTestSavedModelPath);

  Status status = session_bundle::LoadSessionBundleOrSavedModelBundle(
      session_options, run_options, export_dir, tags, &bundle,
      &is_session_bundle);
  EXPECT_TRUE(status.ok());
}

TEST(SessionBundleTest, LoadSessionBundleOrSavedModelBundleFailureTest) {
  SessionOptions session_options;
  RunOptions run_options;
  SavedModelBundle bundle;
  bool is_session_bundle;
  const std::unordered_set<string> tags = {"serve"};
  const string export_dir =
      test_util::TestSrcDirPath(kTestSessionBundleExportPath);

  Status status = session_bundle::LoadSessionBundleOrSavedModelBundle(
      session_options, run_options, export_dir, tags, &bundle,
      &is_session_bundle);
  EXPECT_EQ(::tensorflow::error::UNIMPLEMENTED, status.code());
  EXPECT_THAT(
      status.ToString(),
      ::testing::HasSubstr("Session Bundle is deprecated and removed."));
}

TEST(SessionBundleTest, LoadSessionBundleFromPathUsingRunOptionsTest) {
  SessionOptions session_options;
  RunOptions run_options;
  string export_dir = "/exort_dir";
  SessionBundle bundle;
  Status status = session_bundle::LoadSessionBundleFromPathUsingRunOptions(
      session_options, run_options, export_dir, &bundle);
  EXPECT_EQ(::tensorflow::error::UNIMPLEMENTED, status.code());
  EXPECT_THAT(
      status.ToString(),
      ::testing::HasSubstr("Session Bundle is deprecated and removed."));
}

TEST(SessionBundleTest, SetSignaturesTest) {
  Signatures signatures;
  tensorflow::MetaGraphDef meta_graph_def;
  Status status = session_bundle::SetSignatures(signatures, &meta_graph_def);
  EXPECT_EQ(::tensorflow::error::UNIMPLEMENTED, status.code());
  EXPECT_THAT(
      status.ToString(),
      ::testing::HasSubstr("Session Bundle is deprecated and removed."));
}

TEST(SessionBundleTest, GetClassificationSignatureTest) {
  ClassificationSignature signature;
  tensorflow::MetaGraphDef meta_graph_def;
  Status status =
      session_bundle::GetClassificationSignature(meta_graph_def, &signature);
  EXPECT_EQ(::tensorflow::error::UNIMPLEMENTED, status.code());
  EXPECT_THAT(
      status.ToString(),
      ::testing::HasSubstr("Session Bundle is deprecated and removed."));
}

TEST(SessionBundleTest, GetRegressionSignatureTest) {
  RegressionSignature signature;
  tensorflow::MetaGraphDef meta_graph_def;
  Status status =
      session_bundle::GetRegressionSignature(meta_graph_def, &signature);
  EXPECT_EQ(::tensorflow::error::UNIMPLEMENTED, status.code());
  EXPECT_THAT(
      status.ToString(),
      ::testing::HasSubstr("Session Bundle is deprecated and removed."));
}

TEST(SessionBundleTest, RunClassificationTest) {
  ClassificationSignature signature;
  Tensor input;
  Session* session = nullptr;
  Tensor classes;
  Tensor scores;
  Status status = session_bundle::RunClassification(signature, input, session,
                                                    &classes, &scores);
  EXPECT_EQ(::tensorflow::error::UNIMPLEMENTED, status.code());
  EXPECT_THAT(
      status.ToString(),
      ::testing::HasSubstr("Session Bundle is deprecated and removed."));
}

TEST(SessionBundleTest, RunRegressionTest) {
  RegressionSignature signature;
  Tensor input, output;
  Session* session = nullptr;
  Status status =
      session_bundle::RunRegression(signature, input, session, &output);
  EXPECT_EQ(::tensorflow::error::UNIMPLEMENTED, status.code());
  EXPECT_THAT(
      status.ToString(),
      ::testing::HasSubstr("Session Bundle is deprecated and removed."));
}

TEST(SessionBundleTest, GetNamedSignature) {
  const string name = "name";
  const tensorflow::MetaGraphDef meta_graph_def;
  Signature default_signature;
  Status status = session_bundle::GetNamedSignature(name, meta_graph_def,
                                                    &default_signature);
  EXPECT_EQ(::tensorflow::error::UNIMPLEMENTED, status.code());
  EXPECT_THAT(
      status.ToString(),
      ::testing::HasSubstr("Session Bundle is deprecated and removed."));
}

}  // namespace
}  // namespace session_bundle
}  // namespace serving
}  // namespace tensorflow
