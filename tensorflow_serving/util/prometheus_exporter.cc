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

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "re2/re2.h"

namespace tensorflow {
namespace serving {

namespace {

string SanitizeLabelValue(const string& value) {
  // Backslash and double quote have to be escaped.
  string new_value = value;
  // Replace \ with \\.
  RE2::GlobalReplace(&new_value, "\\\\", "\\\\\\\\");
  // Replace " with \".
  RE2::GlobalReplace(&new_value, "\"", "\\\\\"");
  return new_value;
}

string SanatizeLabelName(const string& name) {
  // Valid format: [a-zA-Z_][a-zA-Z0-9_]*
  string new_name = name;
  RE2::GlobalReplace(&new_name, "[^a-zA-Z0-9]", "_");
  if (RE2::FullMatch(new_name, "^[0-9].*")) {
    // Start with 0-9, prepend a underscore.
    new_name = absl::StrCat("_", new_name);
  }
  return new_name;
}

string GetPrometheusMetricName(
    const monitoring::MetricDescriptor& metric_descriptor) {
  // Valid format: [a-zA-Z_:][a-zA-Z0-9_:]*
  string new_name = metric_descriptor.name;
  RE2::GlobalReplace(&new_name, "[^a-zA-Z0-9_]", ":");
  if (RE2::FullMatch(new_name, "^[0-9].*")) {
    // Start with 0-9, prepend a underscore.
    new_name = absl::StrCat("_", new_name);
  }
  return new_name;
}

void SerializeHistogram(const monitoring::MetricDescriptor& metric_descriptor,
                        const monitoring::PointSet& point_set,
                        std::vector<string>* lines) {
  // For a metric name NAME, we should output:
  //   NAME_bucket{le=b1} x1
  //   NAME_bucket{le=b2} x2
  //   NAME_bucket{le=b3} x3 ...
  //   NAME_sum xsum
  //   NAME_count xcount
  string prom_metric_name = GetPrometheusMetricName(metric_descriptor);
  // Type definition line.
  lines->push_back(absl::StrFormat("# TYPE %s histogram", prom_metric_name));
  for (const auto& point : point_set.points) {
    // Each points has differnet label values.
    std::vector<string> labels = {};
    labels.reserve(point->labels.size());
    for (const auto& label : point->labels) {
      labels.push_back(absl::StrFormat("%s=\"%s\"",
                                       SanatizeLabelName(label.name),
                                       SanitizeLabelValue(label.value)));
    }
    int64 cumulative_count = 0;
    string bucket_prefix =
        absl::StrCat(prom_metric_name, "_bucket{", absl::StrJoin(labels, ","));
    if (!labels.empty()) {
      absl::StrAppend(&bucket_prefix, ",");
    }
    // One bucket per line, last one should be le="Inf".
    for (int i = 0; i < point->histogram_value.bucket_size(); i++) {
      cumulative_count += point->histogram_value.bucket(i);
      string bucket_limit =
          (i < point->histogram_value.bucket_size() - 1)
              ? absl::StrCat(point->histogram_value.bucket_limit(i))
              : "+Inf";
      lines->push_back(absl::StrCat(
          bucket_prefix, absl::StrFormat("le=\"%s\"} ", bucket_limit),
          cumulative_count));
    }
    // _sum and _count.
    lines->push_back(absl::StrCat(prom_metric_name, "_sum{",
                                  absl::StrJoin(labels, ","), "} ",
                                  point->histogram_value.sum()));
    lines->push_back(absl::StrCat(prom_metric_name, "_count{",
                                  absl::StrJoin(labels, ","), "} ",
                                  cumulative_count));
  }
}

void SerializeScalar(const monitoring::MetricDescriptor& metric_descriptor,
                     const monitoring::PointSet& point_set,
                     std::vector<string>* lines) {
  // A counter or gauge metric.
  // The format should be:
  //   NAME{label=value,label=value} x time
  string prom_metric_name = GetPrometheusMetricName(metric_descriptor);
  string metric_type_str = "untyped";
  if (metric_descriptor.metric_kind == monitoring::MetricKind::kCumulative) {
    metric_type_str = "counter";
  } else if (metric_descriptor.metric_kind == monitoring::MetricKind::kGauge) {
    metric_type_str = "gauge";
  }
  // Type definition line.
  lines->push_back(
      absl::StrFormat("# TYPE %s %s", prom_metric_name, metric_type_str));
  for (const auto& point : point_set.points) {
    // Each points has differnet label values.
    string name_bracket = absl::StrCat(prom_metric_name, "{");
    std::vector<string> labels = {};
    labels.reserve(point->labels.size());
    for (const auto& label : point->labels) {
      labels.push_back(absl::StrFormat("%s=\"%s\"",
                                       SanatizeLabelName(label.name),
                                       SanitizeLabelValue(label.value)));
    }
    lines->push_back(absl::StrCat(name_bracket, absl::StrJoin(labels, ","),
                                  absl::StrFormat("} %d", point->int64_value)));
  }
}

void SerializeMetric(const monitoring::MetricDescriptor& metric_descriptor,
                     const monitoring::PointSet& point_set,
                     std::vector<string>* lines) {
  if (metric_descriptor.value_type == monitoring::ValueType::kHistogram) {
    SerializeHistogram(metric_descriptor, point_set, lines);
  } else {
    SerializeScalar(metric_descriptor, point_set, lines);
  }
}

}  // namespace

const char* const PrometheusExporter::kPrometheusPath =
    "/monitoring/prometheus/metrics";

PrometheusExporter::PrometheusExporter()
    : collection_registry_(monitoring::CollectionRegistry::Default()) {}

Status PrometheusExporter::GeneratePage(string* http_page) {
  if (http_page == nullptr) {
    return Status(error::Code::INVALID_ARGUMENT, "Http page pointer is null");
  }
  monitoring::CollectionRegistry::CollectMetricsOptions collect_options;
  collect_options.collect_metric_descriptors = true;
  const std::unique_ptr<monitoring::CollectedMetrics> collected_metrics =
      collection_registry_->CollectMetrics(collect_options);

  const auto& descriptor_map = collected_metrics->metric_descriptor_map;
  const auto& metric_map = collected_metrics->point_set_map;

  std::vector<string> lines;
  for (const auto& name_and_metric_descriptor : descriptor_map) {
    const string& metric_name = name_and_metric_descriptor.first;
    auto metric_iterator = metric_map.find(metric_name);
    if (metric_iterator == metric_map.end()) {
      // Not found.
      continue;
    }
    SerializeMetric(*name_and_metric_descriptor.second,
                    *(metric_iterator->second), &lines);
  }
  *http_page = absl::StrJoin(lines, "\n");
  absl::StrAppend(http_page, "\n");
  return Status::OK();
}

}  // namespace serving
}  // namespace tensorflow
