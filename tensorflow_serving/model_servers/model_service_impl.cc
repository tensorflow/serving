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

#include "tensorflow_serving/model_servers/model_service_impl.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "xla/tsl/lib/monitoring/collected_metrics.h"
#include "xla/tsl/lib/monitoring/collection_registry.h"
#include "tensorflow_serving/model_servers/get_model_status_impl.h"
#include "tensorflow_serving/model_servers/grpc_status_util.h"
#include "tensorflow_serving/util/status_util.h"

namespace tensorflow {
namespace serving {

::grpc::Status ModelServiceImpl::GetModelStatus(
    ::grpc::ServerContext *context, const GetModelStatusRequest *request,
    GetModelStatusResponse *response) {
  const ::grpc::Status status = tensorflow::serving::ToGRPCStatus(
      GetModelStatusImpl::GetModelStatus(core_, *request, response));
  if (!status.ok()) {
    VLOG(1) << "GetModelStatus failed: " << status.error_message();
  }
  return status;
}

::grpc::Status ModelServiceImpl::HandleReloadConfigRequest(
    ::grpc::ServerContext *context, const ReloadConfigRequest *request,
    ReloadConfigResponse *response) {
  ModelServerConfig server_config = request->config();
  Status status;
  const absl::flat_hash_map<std::string, int64_t> old_metric_values =
      GetMetrics(request);
  switch (server_config.config_case()) {
    case ModelServerConfig::kModelConfigList: {
      const ModelConfigList list = server_config.model_config_list();

      for (int index = 0; index < list.config_size(); index++) {
        const ModelConfig config = list.config(index);
        LOG(INFO) << "\nConfig entry"
                  << "\n\tindex : " << index
                  << "\n\tpath : " << config.base_path()
                  << "\n\tname : " << config.name()
                  << "\n\tplatform : " << config.model_platform();
      }
      status = core_->ReloadConfig(server_config);
      break;
    }
    default:
      status = errors::InvalidArgument(
          "ServerModelConfig type not supported by HandleReloadConfigRequest."
          " Only ModelConfigList is currently supported");
  }

  if (!status.ok()) {
    LOG(ERROR) << "ReloadConfig failed: " << status.message();
  }
  const absl::flat_hash_map<std::string, int64_t> new_metric_values =
      GetMetrics(request);
  RecordMetricsIncrease(old_metric_values, new_metric_values, response);

  const StatusProto status_proto = ToStatusProto(status);
  *response->mutable_status() = status_proto;
  return ToGRPCStatus(status);
}

absl::flat_hash_map<std::string, int64_t> ModelServiceImpl::GetMetrics(
    const ReloadConfigRequest *request) {
  absl::flat_hash_map<std::string, int64_t> metric_values = {};
  const tsl::monitoring::CollectionRegistry::CollectMetricsOptions options;
  tsl::monitoring::CollectionRegistry *collection_registry =
      tsl::monitoring::CollectionRegistry::Default();
  std::unique_ptr<tsl::monitoring::CollectedMetrics> collected_metrics =
      collection_registry->CollectMetrics(options);

  for (const std::string &metric_name : request->metric_names()) {
    int64_t metric_value = 0;
    auto it = collected_metrics->point_set_map.find(metric_name);
    if (it != collected_metrics->point_set_map.end()) {
      std::vector<std::unique_ptr<tsl::monitoring::Point>> *points =
          &it->second->points;
      if (!points->empty()) {
        metric_value = (*points)[0]->int64_value;
      }
    }
    metric_values.insert({metric_name, metric_value});
  }
  return metric_values;
}

void ModelServiceImpl::RecordMetricsIncrease(
    const absl::flat_hash_map<std::string, int64_t> &old_metric_values,
    const absl::flat_hash_map<std::string, int64_t> &new_metric_values,
    ReloadConfigResponse *response) {
  for (const auto &[metric_name, metric_value] : new_metric_values) {
    Metric metric;
    metric.set_name(metric_name);
    int64_t old_metric_value = old_metric_values.contains(metric_name)
                                   ? old_metric_values.at(metric_name)
                                   : 0;
    metric.set_int64_value_increase(metric_value - old_metric_value);
    *response->add_metric() = metric;
  }
}
}  // namespace serving
}  // namespace tensorflow
