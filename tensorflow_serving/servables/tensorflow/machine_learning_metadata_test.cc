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

bool GetMlmdUuid(const string& model_name, const string& version,
                 std::string* mlmd_uuid) {
  auto* collection_registry = tsl::monitoring::CollectionRegistry::Default();
  tsl::monitoring::CollectionRegistry::CollectMetricsOptions options;
  const std::unique_ptr<tsl::monitoring::CollectedMetrics> collected_metrics =
      collection_registry->CollectMetrics(options);
  const auto& point_set_map = collected_metrics->point_set_map;
  if (point_set_map.empty() ||
      point_set_map.find(mlmd_streamz) == point_set_map.end())
    return false;
  const tsl::monitoring::PointSet& lps =
      *collected_metrics->point_set_map.at(mlmd_streamz);
  for (int i = 0; i < lps.points.size(); ++i) {
    if ((lps.points[i]->labels[0].name == "model_name") &&
        (lps.points[i]->labels[0].value == model_name) &&
        (lps.points[i]->labels[1].name == "version") &&
        (lps.points[i]->labels[1].value == version)) {
      *mlmd_uuid = lps.points[i]->string_value;
      return true;
    }
  }
  return false;
}

TEST(MachineLearningMetaDataTest, BasicTest_MLMD_missing) {
  std::string mlmd_uuid;
  ASSERT_FALSE(GetMlmdUuid("missing_model", "9696", &mlmd_uuid));
  string test_data_path = test_util::GetTestSavedModelPath();
  MaybePublishMLMDStreamz(test_data_path, "missing_model", 9696);
  EXPECT_FALSE(GetMlmdUuid("missing_model", "9696", &mlmd_uuid));
}

TEST(MachineLearningMetaDataTest, BasicTest_MLMD_present) {
  std::string mlmd_uuid;
  ASSERT_FALSE(GetMlmdUuid("test_model", "9696", &mlmd_uuid));
  const string test_data_path = test_util::TestSrcDirPath(
      strings::StrCat("/servables/tensorflow/testdata/",
                      "saved_model_half_plus_two_mlmd/00000123"));
  MaybePublishMLMDStreamz(test_data_path, "test_model", 9696);
  EXPECT_TRUE(GetMlmdUuid("test_model", "9696", &mlmd_uuid));
  EXPECT_EQ("test_mlmd_uuid", mlmd_uuid);
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
