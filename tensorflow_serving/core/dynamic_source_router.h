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

#ifndef TENSORFLOW_SERVING_CORE_DYNAMIC_SOURCE_ROUTER_H_
#define TENSORFLOW_SERVING_CORE_DYNAMIC_SOURCE_ROUTER_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/platform/macros.h"
#include "tensorflow_serving/core/source_router.h"

namespace tensorflow {
namespace serving {

// A SourceRouter with a fixed set of N output ports, and a dynamically
// reconfigurable map from servable name to port. The route map can only
// reference the first N-1 ports; the Nth port is reserved for servables not
// found in the route map.
template <typename T>
class DynamicSourceRouter final : public SourceRouter<T> {
 public:
  // A servable name -> output port map.
  using Routes = std::map<string, int>;

  // Creates a DynamicSourceRouter with 'num_output_ports' output ports and an
  // (initial) route map given by 'routes'.
  static Status Create(int num_output_ports, const Routes& routes,
                       std::unique_ptr<DynamicSourceRouter<T>>* result);
  ~DynamicSourceRouter() override;

  // Gets the current route map.
  Routes GetRoutes() const;

  // Sets the route map to 'routes'.
  Status UpdateRoutes(const Routes& routes);

 protected:
  int num_output_ports() const override { return num_output_ports_; }

  int Route(const StringPiece servable_name,
            const std::vector<ServableData<T>>& versions) override;

 private:
  DynamicSourceRouter(int num_output_ports, const Routes& routes);

  // Returns an error if 'routes' is invalid, given 'num_output_ports'.
  static Status ValidateRoutes(int num_output_ports, const Routes& routes);

  const int num_output_ports_;

  mutable mutex routes_mu_;
  Routes routes_ GUARDED_BY(routes_mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(DynamicSourceRouter);
};

//////////
// Implementation details follow. API users need not read.

template <typename T>
Status DynamicSourceRouter<T>::Create(
    int num_output_ports, const Routes& routes,
    std::unique_ptr<DynamicSourceRouter<T>>* result) {
  TF_RETURN_IF_ERROR(ValidateRoutes(num_output_ports, routes));
  result->reset(new DynamicSourceRouter<T>(num_output_ports, routes));
  return Status::OK();
}

template <typename T>
DynamicSourceRouter<T>::~DynamicSourceRouter() {
  TargetBase<T>::Detach();
}

template <typename T>
typename DynamicSourceRouter<T>::Routes DynamicSourceRouter<T>::GetRoutes()
    const {
  mutex_lock l(routes_mu_);
  return routes_;
}

template <typename T>
Status DynamicSourceRouter<T>::UpdateRoutes(const Routes& routes) {
  TF_RETURN_IF_ERROR(ValidateRoutes(num_output_ports_, routes));
  {
    mutex_lock l(routes_mu_);
    routes_ = routes;
  }
  return Status::OK();
}

template <typename T>
int DynamicSourceRouter<T>::Route(
    const StringPiece servable_name,
    const std::vector<ServableData<T>>& versions) {
  mutex_lock l(routes_mu_);
  auto it = routes_.find(servable_name.ToString());
  if (it == routes_.end()) {
    LOG(INFO) << "Routing servable(s) from stream " << servable_name
              << " to default output port " << num_output_ports_ - 1;
    return num_output_ports_ - 1;
  } else {
    return it->second;
  }
}

template <typename T>
DynamicSourceRouter<T>::DynamicSourceRouter(int num_output_ports,
                                            const Routes& routes)
    : num_output_ports_(num_output_ports), routes_(routes) {}

template <typename T>
Status DynamicSourceRouter<T>::ValidateRoutes(int num_output_ports,
                                              const Routes& routes) {
  for (const auto& entry : routes) {
    const int port = entry.second;
    if (port < 0 || port >= num_output_ports) {
      return errors::InvalidArgument(
          strings::StrCat("Port number out of range: ", port));
    }
    if (port == num_output_ports - 1) {
      return errors::InvalidArgument(
          "Last port cannot be used in route map, since it's reserved for the "
          "default route");
    }
  }
  return Status::OK();
}

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_DYNAMIC_SOURCE_ROUTER_H_
