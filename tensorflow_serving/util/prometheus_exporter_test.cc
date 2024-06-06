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

#include "tensorflow_serving/util/prometheus_exporter.h"

#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow/core/lib/monitoring/sampler.h"

namespace tensorflow {
namespace serving {
namespace {

TEST(PrometheusExporterTest, Counter) {
  auto exporter = absl::make_unique<PrometheusExporter>();
  auto counter = absl::WrapUnique(
      monitoring::Counter<1>::New("/test/path/total", "A counter.", "name"));
  counter->GetCell("abc")->IncrementBy(2);

  string http_page;
  Status status = exporter->GeneratePage(&http_page);

  string expected_result = absl::StrJoin(
      {"# TYPE :test:path:total counter", ":test:path:total{name=\"abc\"} 2"},
      "\n");
  absl::StrAppend(&expected_result, "\n");
  EXPECT_PRED_FORMAT2(testing::IsSubstring, expected_result, http_page);
}

TEST(PrometheusExporterTest, Gauge) {
  auto exporter = absl::make_unique<PrometheusExporter>();
  auto gauge = absl::WrapUnique(monitoring::Gauge<int64_t, 2>::New(
      "/test/path/gague", "A gauge", "x", "y"));
  gauge->GetCell("abc", "def")->Set(5);

  string http_page;
  Status status = exporter->GeneratePage(&http_page);
  string expected_result =
      absl::StrJoin({"# TYPE :test:path:gague gauge",
                     ":test:path:gague{x=\"abc\",y=\"def\"} 5"},
                    "\n");
  absl::StrAppend(&expected_result, "\n");
  EXPECT_PRED_FORMAT2(testing::IsSubstring, expected_result, http_page);
}

TEST(PrometheusExporterTest, Histogram) {
  auto exporter = absl::make_unique<PrometheusExporter>();
  auto histogram = absl::WrapUnique(monitoring::Sampler<1>::New(
      {"/test/path/histogram", "A histogram.", "status"},
      monitoring::Buckets::Exponential(1, 2, 10)));
  histogram->GetCell("good")->Add(2);
  histogram->GetCell("good")->Add(20);
  histogram->GetCell("good")->Add(200);

  string http_page;
  Status status = exporter->GeneratePage(&http_page);
  string expected_result = absl::StrJoin(
      {"# TYPE :test:path:histogram histogram",
       ":test:path:histogram_bucket{status=\"good\",le=\"1\"} 0",
       ":test:path:histogram_bucket{status=\"good\",le=\"2\"} 0",
       ":test:path:histogram_bucket{status=\"good\",le=\"4\"} 1",
       ":test:path:histogram_bucket{status=\"good\",le=\"8\"} 1",
       ":test:path:histogram_bucket{status=\"good\",le=\"16\"} 1",
       ":test:path:histogram_bucket{status=\"good\",le=\"32\"} 2",
       ":test:path:histogram_bucket{status=\"good\",le=\"64\"} 2",
       ":test:path:histogram_bucket{status=\"good\",le=\"128\"} 2",
       ":test:path:histogram_bucket{status=\"good\",le=\"256\"} 3",
       ":test:path:histogram_bucket{status=\"good\",le=\"512\"} 3",
       ":test:path:histogram_bucket{status=\"good\",le=\"+Inf\"} 3",
       ":test:path:histogram_sum{status=\"good\"} 222",
       ":test:path:histogram_count{status=\"good\"} 3"},
      "\n");
  absl::StrAppend(&expected_result, "\n");
  EXPECT_PRED_FORMAT2(testing::IsSubstring, expected_result, http_page);
}

TEST(PrometheusExporterTest, SanitizeLabelValue) {
  auto exporter = absl::make_unique<PrometheusExporter>();
  auto counter = absl::WrapUnique(
      monitoring::Counter<1>::New("/test/path/total", "A counter.", "name"));
  // label value: "abc\"
  counter->GetCell("\"abc\\\"")->IncrementBy(2);

  string http_page;
  Status status = exporter->GeneratePage(&http_page);

  string expected_result =
      absl::StrJoin({"# TYPE :test:path:total counter",
                     ":test:path:total{name=\"\\\"abc\\\\\\\"\"} 2"},
                    "\n");
  absl::StrAppend(&expected_result, "\n");
  EXPECT_PRED_FORMAT2(testing::IsSubstring, expected_result, http_page);
}

TEST(PrometheusExporterTest, SanitizeLabelName) {
  auto exporter = absl::make_unique<PrometheusExporter>();
  auto counter = absl::WrapUnique(monitoring::Counter<1>::New(
      "/test/path/total", "A counter.", "my-name+1"));
  counter->GetCell("abc")->IncrementBy(2);

  string http_page;
  Status status = exporter->GeneratePage(&http_page);

  string expected_result =
      absl::StrJoin({"# TYPE :test:path:total counter",
                     ":test:path:total{my_name_1=\"abc\"} 2"},
                    "\n");
  absl::StrAppend(&expected_result, "\n");
  EXPECT_PRED_FORMAT2(testing::IsSubstring, expected_result, http_page);
}

TEST(PrometheusExporterTest, SanitizeMetricName) {
  auto exporter = absl::make_unique<PrometheusExporter>();
  auto counter = absl::WrapUnique(
      monitoring::Counter<1>::New("0/path-total_count", "A counter.", "name"));
  counter->GetCell("abc")->IncrementBy(2);

  string http_page;
  Status status = exporter->GeneratePage(&http_page);

  string expected_result =
      absl::StrJoin({"# TYPE _0:path:total_count counter",
                     "_0:path:total_count{name=\"abc\"} 2"},
                    "\n");
  absl::StrAppend(&expected_result, "\n");
  EXPECT_PRED_FORMAT2(testing::IsSubstring, expected_result, http_page);
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
