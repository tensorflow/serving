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

#include "tensorflow_serving/servables/tensorflow/machine_learning_metadata.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow_serving/servables/tensorflow/bundle_factory_test_util.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace {

const char mlmd_streamz[] = "/tensorflow/serving/mlmd_map";

class MachineLearningMetaDataTest : public ::testing::Test {
 protected:
  MachineLearningMetaDataTest() {}
  virtual ~MachineLearningMetaDataTest() = default;
};

TEST_F(MachineLearningMetaDataTest, BasicTest) {
  // Keep both cases in the same test as there's no way to reset the TFStreamz
  // between reads. This leads to ordering/sharding dependencies on the test.
  {
    // No MLMD in SavedModel case.
    string test_data_path = test_util::GetTestSavedModelPath();
    MaybePublishMLMDStreamz(test_data_path, "missing_model", 9696);
    auto* collection_registry = monitoring::CollectionRegistry::Default();
    monitoring::CollectionRegistry::CollectMetricsOptions options;
    const std::unique_ptr<monitoring::CollectedMetrics> collected_metrics =
        collection_registry->CollectMetrics(options);
    const monitoring::PointSet& lps =
        *collected_metrics->point_set_map.at(mlmd_streamz);
    EXPECT_EQ(0, lps.points.size());
  }
  {
    // MLMD in SavedModel case.
    const string test_data_path = test_util::TestSrcDirPath(
        strings::StrCat("/servables/tensorflow/testdata/",
                        "saved_model_half_plus_two_mlmd/00000123"));
    MaybePublishMLMDStreamz(test_data_path, "test_model", 9696);
    auto* collection_registry = monitoring::CollectionRegistry::Default();
    monitoring::CollectionRegistry::CollectMetricsOptions options;
    const std::unique_ptr<monitoring::CollectedMetrics> collected_metrics =
        collection_registry->CollectMetrics(options);
    const monitoring::PointSet& lps =
        *collected_metrics->point_set_map.at(mlmd_streamz);
    EXPECT_EQ(1, lps.points.size());
    EXPECT_EQ(2, lps.points[0]->labels.size());
    EXPECT_EQ("model_name", lps.points[0]->labels[0].name);
    EXPECT_EQ("test_model", lps.points[0]->labels[0].value);
    EXPECT_EQ("version", lps.points[0]->labels[1].name);
    EXPECT_EQ("9696", lps.points[0]->labels[1].value);
    EXPECT_EQ("test_mlmd_uuid", lps.points[0]->string_value);
  }
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
