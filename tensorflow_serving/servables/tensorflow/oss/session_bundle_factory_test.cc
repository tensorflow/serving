/* Copyright 2016 Google Inc. All Rights Reserved.

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

#include "tensorflow_serving/servables/tensorflow/session_bundle_factory.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/wrappers.pb.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow_serving/servables/tensorflow/bundle_factory_test.h"
#include "tensorflow_serving/servables/tensorflow/bundle_factory_test_util.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_config.pb.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace {

class SessionBundleFactoryTest : public test_util::BundleFactoryTest {
 public:
  SessionBundleFactoryTest()
      : test_util::BundleFactoryTest(
            test_util::GetTestSessionBundleExportPath()) {}

  virtual ~SessionBundleFactoryTest() = default;

  Status CreateSession(const SessionBundleConfig& config,
                       std::unique_ptr<Session>* session) const override {
    std::unique_ptr<SessionBundleFactory> factory;
    TF_RETURN_IF_ERROR(SessionBundleFactory::Create(config, &factory));
    std::unique_ptr<SessionBundle> bundle;
    TF_RETURN_IF_ERROR(factory->CreateSessionBundle(export_dir_, &bundle));
    *session = std::move(bundle->session);
    return Status::OK();
  }
};

TEST_F(SessionBundleFactoryTest, Basic) {
  const SessionBundleConfig config;
  std::unique_ptr<Session> session;
  Status status = CreateSession(config, &session);
  EXPECT_EQ(::tensorflow::error::UNIMPLEMENTED, status.code());
  EXPECT_THAT(
      status.ToString(),
      ::testing::HasSubstr("Session Bundle is deprecated and removed."));
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
