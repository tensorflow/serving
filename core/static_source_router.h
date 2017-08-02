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

#ifndef TENSORFLOW_SERVING_CORE_STATIC_SOURCE_ROUTER_H_
#define TENSORFLOW_SERVING_CORE_STATIC_SOURCE_ROUTER_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/platform/macros.h"
#include "tensorflow_serving/core/source_router.h"

namespace tensorflow {
namespace serving {

// A SourceRouter with N statically-configured output ports. Items are routed to
// output ports based on substring matching against the servable name. The
// router is configured with N-1 substrings, with "fall-through" semantics. In
// particular: The substrings are numbered 0, 1, ..., N-2. Items whose servable
// name matches substring 0 are sent to port 0; items that fail to match
// substring 0 but do match substring 1 are sent to port 1; and so on. Items
// that match none of the substrings are sent to port N-1.
template <typename T>
class StaticSourceRouter final : public SourceRouter<T> {
 public:
  // Creates a StaticSourceRouter with 'route_substrings.size() + 1' output
  // ports, based on cascading substring matching as described above.
  static Status Create(const std::vector<string>& route_substrings,
                       std::unique_ptr<StaticSourceRouter<T>>* result);
  ~StaticSourceRouter() override;

 protected:
  int num_output_ports() const override {
    return routes_except_default_.size() + 1;
  }

  int Route(const StringPiece servable_name,
            const std::vector<ServableData<T>>& versions) override;

 private:
  explicit StaticSourceRouter(const std::vector<string>& route_substrings);

  // The substrings of the first N-1 routes (the Nth route is the default
  // route).
  std::vector<string> routes_except_default_;

  TF_DISALLOW_COPY_AND_ASSIGN(StaticSourceRouter);
};

//////////
// Implementation details follow. API users need not read.

template <typename T>
Status StaticSourceRouter<T>::Create(
    const std::vector<string>& route_substrings,
    std::unique_ptr<StaticSourceRouter<T>>* result) {
  result->reset(new StaticSourceRouter<T>(route_substrings));
  return Status::OK();
}

template <typename T>
StaticSourceRouter<T>::~StaticSourceRouter() {
  TargetBase<T>::Detach();
}

template <typename T>
int StaticSourceRouter<T>::Route(const StringPiece servable_name,
                                 const std::vector<ServableData<T>>& versions) {
  for (int i = 0; i < routes_except_default_.size(); ++i) {
    if (servable_name.contains(routes_except_default_[i])) {
      LOG(INFO) << "Routing servable(s) from stream " << servable_name
                << " to route " << i;
      return i;
    }
  }
  // None of the substrings matched, so return the "default" Nth route.
  LOG(INFO) << "Routing servable(s) from stream " << servable_name
            << " to default route " << routes_except_default_.size();
  return routes_except_default_.size();
}

template <typename T>
StaticSourceRouter<T>::StaticSourceRouter(
    const std::vector<string>& route_substrings)
    : routes_except_default_(route_substrings) {}

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_STATIC_SOURCE_ROUTER_H_
