#ifndef TENSORFLOW_SERVING_CORE_SERVABLE_METRICS_MONITOR_H_
#define TENSORFLOW_SERVING_CORE_SERVABLE_METRICS_MONITOR_H_

#include "tensorflow_serving/core/servable_metrics.h"
#include "tensorflow_serving/core/servable_state_monitor.h"

namespace tensorflow {
namespace serving {

/// A monitor that manage servable metrics by listing to EventBus<ServableState>.
/// Manage metrics for each model name, version and MethodType.
class ServableMetricsMonitor {
 public:
  enum class MethodType : int {
    kUnknown,
    kClassify,
    kRegress,
    kPredict,
  };

  using MethodTypeMetricsMap =
      std::map<MethodType, ServableMetrics>;

  using VersionMethodTypeMetricsMap =
      std::map<ServableStateMonitor::Version, MethodTypeMetricsMap>;

  using ServableMetricsMap =
      std::map<ServableStateMonitor::ServableName, VersionMethodTypeMetricsMap>;

  ServableMetricsMonitor() = default;
  virtual ~ServableMetricsMonitor() = default;

  /// Create a notifier to handle state to measure metrics.
  virtual ServableStateMonitor::ServableStateNotifierFn CreateNotifier(
          const ServableId& servable_id,
          const ServableMetricsMonitor::MethodType& method_type,
          const ServableMetricsRequest& request);

  /// Returns the current metrics of one servable, or nullopt if that servable is
  /// not being tracked.
  optional<ServableMetricsMonitor::MethodTypeMetricsMap>
  GetMethodTypeMetrics(const ServableId& servable_id) const LOCKS_EXCLUDED(mu_);

  /// Returns the current metrics of all tracked versions of the given servable,
  /// if any.
  optional<ServableMetricsMonitor::VersionMethodTypeMetricsMap>
  GetVersionMethodTypeMetrics(const string& servable_name) const LOCKS_EXCLUDED(mu_);

 private:
  mutable mutex mu_;

  void UpdateMetrics(const ServableId& servable_id,
                     const ServableMetricsMonitor::MethodType& method_type,
                     const ServableMetricsRequest& request) LOCKS_EXCLUDED(mu_);

  ServableMetricsMonitor::ServableMetricsMap metrics_ GUARDED_BY(mu_);
};

}  // namespace serving
}  // namespace tensorflow

#endif //TENSORFLOW_SERVING_CORE_SERVABLE_METRICS_MONITOR_H_
