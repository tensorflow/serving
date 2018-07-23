#include "tensorflow_serving/core/servable_metrics_monitor.h"

namespace tensorflow {
namespace serving {

ServableStateMonitor::ServableStateNotifierFn
ServableMetricsMonitor::CreateNotifier(
        const ServableId& servable_id, const ServableMetricsMonitor::MethodType& method_type,
        const ServableMetricsRequest& request) {

  const ServableStateMonitor::ServableStateNotifierFn& notifier_fn =
  [&](const bool reached,
      std::map<ServableId, ServableState::ManagerState> states_reached) {
    for (auto state_reached : states_reached) {
      switch (state_reached.second) {
        case ServableState::ManagerState::kStart:
        case ServableState::ManagerState::kUnloading:
        case ServableState::ManagerState::kAvailable:
        case ServableState::ManagerState::kLoading:
          break;
        case ServableState::ManagerState::kEnd:
          ServableMetricsMonitor::UpdateMetrics(servable_id, method_type, request);
          auto metrics = ServableMetricsMonitor::GetMethodTypeMetrics(servable_id).value()[method_type];
          break;
      }
    }
  };
  return notifier_fn;
}

void ServableMetricsMonitor::UpdateMetrics(
        const ServableId& servable_id, const ServableMetricsMonitor::MethodType& method_type,
        const ServableMetricsRequest& request) {
  mutex_lock l(mu_);
  metrics_[servable_id.name][servable_id.version][method_type].request_count += request.request_value;
  metrics_[servable_id.name][servable_id.version][method_type].error_count += request.error_value;
}

optional<ServableMetricsMonitor::MethodTypeMetricsMap>
ServableMetricsMonitor::GetMethodTypeMetrics(const ServableId& servable_id) const {
  mutex_lock l(mu_);
  auto it = metrics_.find(servable_id.name);
  if (it == metrics_.end()) {
    return nullopt;
  }

  const VersionMethodTypeMetricsMap& versions = it->second;
  auto it2 = versions.find(servable_id.version);
  if (it2 == versions.end()) {
    return nullopt;
  }

  return it2->second;
}

optional<ServableMetricsMonitor::VersionMethodTypeMetricsMap>
ServableMetricsMonitor::GetVersionMethodTypeMetrics(const string& servable_name) const {
  mutex_lock l(mu_);
  auto it = metrics_.find(servable_name);
  if (it == metrics_.end()) {
    return nullopt;
  }
  return it->second;
}

}  // namespace serving
}  // namespace tensorflow
