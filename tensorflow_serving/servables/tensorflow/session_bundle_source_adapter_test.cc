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

#include "tensorflow_serving/servables/tensorflow/session_bundle_source_adapter.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/contrib/session_bundle/session_bundle.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow_serving/core/loader.h"
#include "tensorflow_serving/core/servable_data.h"
#include "tensorflow_serving/resources/resources.pb.h"
#include "tensorflow_serving/servables/tensorflow/bundle_factory_test_util.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_config.pb.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_source_adapter.pb.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace {

using test_util::EqualsProto;

class SessionBundleSourceAdapterTest : public ::testing::Test {
 protected:
  SessionBundleSourceAdapterTest()
      : export_dir_(test_util::GetTestSessionBundleExportPath()) {}

  // Test data path, to be initialized to point at an export of half-plus-two.
  const string export_dir_;

  void TestSessionBundleSourceAdapter(
      const SessionBundleSourceAdapterConfig& config) const {
    std::unique_ptr<Loader> loader;
    {
      std::unique_ptr<SessionBundleSourceAdapter> adapter;
      TF_CHECK_OK(SessionBundleSourceAdapter::Create(config, &adapter));
      ServableData<std::unique_ptr<Loader>> loader_data =
          adapter->AdaptOneVersion(
              ServableData<StoragePath>({"", 0}, export_dir_));
      TF_ASSERT_OK(loader_data.status());
      loader = loader_data.ConsumeDataOrDie();

      // Let the adapter fall out of scope and be deleted. The loader we got
      // from it should be unaffected. Regression test coverage for b/30202207.
    }

    // We should get a non-empty resource estimate, and we should get the same
    // value twice (via memoization).
    ResourceAllocation first_resource_estimate;
    TF_ASSERT_OK(loader->EstimateResources(&first_resource_estimate));
    EXPECT_FALSE(first_resource_estimate.resource_quantities().empty());
    ResourceAllocation second_resource_estimate;
    TF_ASSERT_OK(loader->EstimateResources(&second_resource_estimate));
    EXPECT_THAT(second_resource_estimate, EqualsProto(first_resource_estimate));

    TF_ASSERT_OK(loader->Load());

    const SessionBundle* bundle = loader->servable().get<SessionBundle>();
    test_util::TestSingleRequest(bundle->session.get());

    loader->Unload();
  }
};

TEST_F(SessionBundleSourceAdapterTest, Basic) {
  const SessionBundleSourceAdapterConfig config;
  TestSessionBundleSourceAdapter(config);
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
