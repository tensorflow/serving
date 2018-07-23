#include "tensorflow_serving/model_servers/get_model_metrics_impl.h"

namespace tensorflow {
namespace serving {
namespace {

ModelVersionMetrics_MethodType MethodTypeToProtoEnum(
  const ServableMetricsMonitor::MethodType& method_type) {
  switch (method_type) {
    case ServableMetricsMonitor::MethodType::kClassify: {
      return ModelVersionMetrics_MethodType_CLASSIFY;
    }
    case ServableMetricsMonitor::MethodType::kRegress: {
      return ModelVersionMetrics_MethodType_REGRESS;
    }
    case ServableMetricsMonitor::MethodType::kPredict: {
      return ModelVersionMetrics_MethodType_PREDICT;
    }
    default:
      return ModelVersionMetrics_MethodType_UNKNOWN;
  }
}

void AddModelVersionMetricsToResponse(GetModelMetricsResponse* response,
                                      const int64& version,
                                      const ServableMetricsMonitor::MethodType& method_type,
                                      const ServableMetrics& servable_metrics) {
  ModelVersionMetrics* version_metrics = response->add_model_version_metrics();
  version_metrics->set_version(version);
  version_metrics->set_method_type(MethodTypeToProtoEnum(method_type));
  Metrics metrics;
  metrics.set_request_count(servable_metrics.request_count);
  metrics.set_error_count(servable_metrics.error_count);
  *version_metrics->mutable_metrics() = metrics;
}

}  // namespace

Status GetModelMetricsImpl::GetModelMetrics(ServerCore* core,
                                            const GetModelMetricsRequest& request,
                                            GetModelMetricsResponse* response) {
  if (!request.has_model_spec()) {
    return tensorflow::errors::InvalidArgument("Missing ModelSpec");
  }

  const string& model_name = request.model_spec().name();
  const ServableMetricsMonitor& monitor = *core->servable_metrics_monitor();

  if (request.model_spec().has_version()) {
    const int64 model_version = request.model_spec().version().value();
    const ServableId servable_id = {model_name, model_version};

    optional<ServableMetricsMonitor::MethodTypeMetricsMap> maybe_method_type_metrics =
        monitor.GetMethodTypeMetrics(servable_id);

    if (!maybe_method_type_metrics) {
      return tensorflow::errors::NotFound("Could not find version", model_version,
                                          " of model ", model_name);
    }
    for (const auto& method_type_metric : maybe_method_type_metrics.value()) {
      const ServableMetricsMonitor::MethodType method_type = method_type_metric.first;
      AddModelVersionMetricsToResponse(response, model_version, method_type, method_type_metric.second);
    }
  } else {
    const optional<ServableMetricsMonitor::VersionMethodTypeMetricsMap> maybe_version_metrics =
        monitor.GetVersionMethodTypeMetrics(model_name);
    if (!maybe_version_metrics) {
      return tensorflow::errors::NotFound(
          "Cound not find any version of model ", model_name);
    }
    for (const auto& version_metric : maybe_version_metrics.value()) {
      const int64 model_version = version_metric.first;
      for (const auto& method_type_metric : version_metric.second) {
        const ServableMetricsMonitor::MethodType method_type = method_type_metric.first;
        AddModelVersionMetricsToResponse(response, model_version, method_type, method_type_metric.second);
      }
    }
  }
  return tensorflow::Status::OK();
}

}  // namespace serving
}  // namespace tensorflow
