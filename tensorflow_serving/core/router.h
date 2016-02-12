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

#ifndef TENSORFLOW_SERVING_CORE_ROUTER_H_
#define TENSORFLOW_SERVING_CORE_ROUTER_H_

#include <algorithm>
#include <memory>
#include <vector>

#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow_serving/core/source.h"
#include "tensorflow_serving/core/source_adapter.h"
#include "tensorflow_serving/core/target.h"

namespace tensorflow {
namespace serving {

// A module that splits aspired-version calls from one input to multiple outputs
// based on some criterion, e.g. the servable name. There is a single input
// "port" represented by Target<T>::SetAspiredVersions(), and N output "ports",
// each of type Source<T>, numbered 0, 1, 2, ...
//
// For a typical use-case, consider a server hosting multiple kinds of servables
// (say, apple servables and orange servables). Perhaps both kinds of servables
// arrive via file-system paths, a la:
//  /path/to/some/apple/servable
//  /path/to/some/orange/servable
// where the servable kinds are distinguished based on the presence of "apple"
// or "orange" in the path. A Router can be interposed between a file-system
// monitoring Source<StoragePath>, and a pair of SourceAdapters (one that emits
// loaders of apple servables, and one that emits loaders of orange servables),
// to route each path to the appropriate SourceAdapter.
template <typename T>
class Router : public TargetBase<T> {
 public:
  ~Router() override = default;

  // Returns a vector of N source pointers, corresponding to the N output ports
  // of the router. The caller must invoke ConnectSourceToTarget() (or directly
  // call SetAspiredVersionsCallback()) on each of them to arrange to route
  // items to various upstream targets. That must be done exactly once, and
  // before calling SetAspiredVersions() on the router.
  std::vector<Source<T>*> GetOutputPorts();

  // Implemented in terms of Route(), defined below and written by the subclass.
  void SetAspiredVersions(const StringPiece servable_name,
                          std::vector<ServableData<T>> versions) final;

 protected:
  // Returns the number of output ports. Must be > 0 and fixed for the lifetime
  // of the router. To be written by the implementing subclass.
  virtual int num_output_ports() const = 0;

  // Returns the output port # to which to route a given aspired-versions
  // request, in [0, num_output_ports() - 1]. To be written by the implementing
  // subclass.
  virtual int Route(const StringPiece servable_name,
                    const std::vector<ServableData<T>>& versions) = 0;

 private:
  // The num_output_ports() output ports. Each one is an IdentitySourceAdapter.
  // Populated in GetOutputPorts().
  std::vector<std::unique_ptr<SourceAdapter<T, T>>> output_ports_;

  // Has 'output_ports_' been populated yet, so that the SourceAdapter is ready
  // to propagate aspired versions?
  Notification output_ports_created_;
};

//////////
// Implementation details follow. API users need not read.

namespace internal {

// A SourceAdapter that passes through data unchanged. Used to implement the
// output ports.
template <typename T>
class IdentitySourceAdapter : public UnarySourceAdapter<T, T> {
 public:
  IdentitySourceAdapter() = default;
  ~IdentitySourceAdapter() override = default;

 protected:
  Status Convert(const T& data, T* converted_data) override {
    *converted_data = data;
    return Status::OK();
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(IdentitySourceAdapter);
};

}  // namespace internal

template <typename T>
std::vector<Source<T>*> Router<T>::GetOutputPorts() {
  if (!output_ports_created_.HasBeenNotified()) {
    int num_ports = num_output_ports();
    if (num_ports < 1) {
      LOG(ERROR) << "Router abstraction used improperly; num_output_ports() "
                    "must return a number greater than 0";
      DCHECK(false);
      num_ports = 1;
    }
    for (int i = 0; i < num_ports; ++i) {
      output_ports_.emplace_back(new internal::IdentitySourceAdapter<T>);
    }
    output_ports_created_.Notify();
  }

  std::vector<Source<T>*> result;
  for (auto& output_port : output_ports_) {
    result.push_back(output_port.get());
  }
  return result;
}

template <typename T>
void Router<T>::SetAspiredVersions(const StringPiece servable_name,
                                   std::vector<ServableData<T>> versions) {
  output_ports_created_.WaitForNotification();
  int output_port = Route(servable_name, versions);
  if (output_port < 0 || output_port > output_ports_.size() - 1) {
    LOG(ERROR) << "Router abstraction used improperly; Route() must return a "
                  "value in [0, num_output_ports()-1]; suppressing the "
                  "aspired-versions request";
    DCHECK(false);
    return;
  }
  output_ports_[output_port]->SetAspiredVersions(servable_name,
                                                 std::move(versions));
}

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_ROUTER_H_
