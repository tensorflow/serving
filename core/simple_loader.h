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

#ifndef TENSORFLOW_SERVING_CORE_SIMPLE_LOADER_H_
#define TENSORFLOW_SERVING_CORE_SIMPLE_LOADER_H_

#include <functional>
#include <memory>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_serving/core/loader.h"
#include "tensorflow_serving/core/source_adapter.h"
#include "tensorflow_serving/resources/resource_values.h"
#include "tensorflow_serving/util/any_ptr.h"
#include "tensorflow_serving/util/optional.h"

namespace tensorflow {
namespace serving {

// SimpleLoader is a wrapper that simplifies Loader creation for common, simple
// use-cases that conform to the following restrictions:
//  - The servable's estimated resource footprint is static.
//  - The servable can be loaded by invoking a no-argument closure.
//  - The servable can be unloaded by invoking its destructor.
//
// When constructing a SimpleLoader users provide a Creator callback. This
// callback is used in the Load() method to construct a servable of type
// ServableType and populate the servable. The servable object is destroyed when
// Unload() is called.
//
// SimpleLoader uses a second supplied callback to estimate the servable's
// resource usage. It memoizes that callback's result, for efficiency. If main-
// memory resources are specified, Unload() releases that amount of memory to
// the OS after deleting the servable.
//
// Example use: create a toy Loader for a servable of type time_t.  Here the
// time servable is instantiated with the current time when Load() is called.
//   auto servable_creator = [](std::unique_ptr<time_t>* servable) {
//       servable->reset(new time_t);
//       *servable = time(nullptr);
//       return Status::OK();
//   };
//   auto resource_estimator = [](ResourceAllocation* estimate) {
//       estimate->mutable_...(...)->set_...(...);
//       return Status::OK();
//   };
//   std::unique_ptr<Loader> loader(new SimpleLoader<time_t>(
//       servable_creator, resource_estimator));
template <typename ServableType>
class SimpleLoader : public Loader {
 public:
  // Creator is called in Load and used to create the servable.
  using Creator = std::function<Status(std::unique_ptr<ServableType>*)>;

  // A callback for estimating a servable's resource usage.
  using ResourceEstimator = std::function<Status(ResourceAllocation*)>;

  // Returns a dummy resource-estimation callback that estimates the servable's
  // resource footprint at zero. Useful in best-effort or test environments that
  // do not track resource usage.
  //
  // IMPORTANT: Use of EstimateNoResources() abdicates resource safety, i.e. a
  // loader using that option does not declare its servable's resource usage,
  // and hence the serving system cannot enforce resource safety.
  static ResourceEstimator EstimateNoResources();

  SimpleLoader(Creator creator, ResourceEstimator resource_estimator);
  ~SimpleLoader() override = default;

  Status EstimateResources(ResourceAllocation* estimate) const override;

  Status Load() override;

  void Unload() override;

  AnyPtr servable() override { return AnyPtr{servable_.get()}; }

 private:
  Creator creator_;

  ResourceEstimator resource_estimator_;

  // The memoized estimated resource requirement of the session bundle servable.
  mutable optional<ResourceAllocation> memoized_resource_estimate_;

  std::unique_ptr<ServableType> servable_;

  TF_DISALLOW_COPY_AND_ASSIGN(SimpleLoader);
};

// SimpleLoaderSourceAdapter is used to create a simple SourceAdapter that
// creates Loaders for servables of type ServableType (e.g. std::map), from
// objects of type DataType (e.g. storage paths of serialized hashmaps).
//
// It bundles together the SourceAdapter and Loader concepts, in a way that
// suffices for simple use cases. In particular, its limitations are:
//
//  - Like UnarySourceAdapter (see source_adapter.h), it translates aspired-
// version items one at a time, giving a simpler interface but less flexibility.
//
//  - Like SimpleLoader, the servable's estimated resource footprint is static,
//    and the emitted loaders' Unload() implementation calls ServableType's
//    destructor and releases the memory to the OS.
//
// For more complex behaviors, SimpleLoaderSourceAdapter is inapplicable. You
// must instead create a SourceAdapter and Loader. That said, you may still be
// able to use one of UnarySourceAdapter or SimpleLoader.
//
// IMPORTANT: Every leaf derived class must call Detach() at the top of its
// destructor. (See documentation on TargetBase::Detach() in target.h.) Doing so
// ensures that no virtual method calls are in flight during destruction of
// member variables.
template <typename DataType, typename ServableType>
class SimpleLoaderSourceAdapter
    : public UnarySourceAdapter<DataType, std::unique_ptr<Loader>> {
 public:
  ~SimpleLoaderSourceAdapter() override = 0;

  // Creator is called by the produced Loaders' Load() method, and used to
  // create objects of type ServableType. It takes a DataType object as input.
  using Creator =
      std::function<Status(const DataType&, std::unique_ptr<ServableType>*)>;

  // A callback for estimating a servable's resource usage. It takes a DataType
  // object as input.
  using ResourceEstimator =
      std::function<Status(const DataType&, ResourceAllocation*)>;

  // Returns a dummy resource-estimation callback that estimates the servable's
  // resource footprint at zero. Useful in best-effort or test environments that
  // do not track resource usage.
  //
  // IMPORTANT: Use of EstimateNoResources() abdicates resource safety, i.e. a
  // loader using that option does not declare its servable's resource usage,
  // and hence the serving system cannot enforce resource safety.
  static ResourceEstimator EstimateNoResources();

 protected:
  // This is an abstract class.
  SimpleLoaderSourceAdapter(Creator creator,
                            ResourceEstimator resource_estimator);

  Status Convert(const DataType& data, std::unique_ptr<Loader>* loader) final;

 private:
  Creator creator_;
  ResourceEstimator resource_estimator_;

  TF_DISALLOW_COPY_AND_ASSIGN(SimpleLoaderSourceAdapter);
};

//////////
// Implementation details follow. API users need not read.

template <typename ServableType>
typename SimpleLoader<ServableType>::ResourceEstimator
SimpleLoader<ServableType>::EstimateNoResources() {
  return [](ResourceAllocation* estimate) {
    estimate->Clear();
    return Status::OK();
  };
}

template <typename ServableType>
SimpleLoader<ServableType>::SimpleLoader(Creator creator,
                                         ResourceEstimator resource_estimator)
    : creator_(creator), resource_estimator_(resource_estimator) {}

template <typename ServableType>
Status SimpleLoader<ServableType>::EstimateResources(
    ResourceAllocation* estimate) const {
  if (memoized_resource_estimate_) {
    *estimate = *memoized_resource_estimate_;
    return Status::OK();
  }

  // Compute and memoize the resource estimate.
  TF_RETURN_IF_ERROR(resource_estimator_(estimate));
  memoized_resource_estimate_ = *estimate;
  return Status::OK();
}

template <typename ServableType>
Status SimpleLoader<ServableType>::Load() {
  const Status status = creator_(&servable_);
  return status;
}

template <typename ServableType>
void SimpleLoader<ServableType>::Unload() {
  // Before destroying the servable, run the resource estimator (in case the
  // estimation routine calls into the servable behind the scenes.)
  ResourceAllocation resource_estimate;
  Status resource_status = EstimateResources(&resource_estimate);

  // Delete the servable no matter what (even if the resource estimator had some
  // error).
  servable_.reset();

  if (!resource_status.ok()) {
    return;
  }

  // If we have a main-memory footprint estimate, release that amount of memory
  // to the OS.
  for (const ResourceAllocation::Entry& entry :
       resource_estimate.resource_quantities()) {
    if (entry.resource().device() == device_types::kMain &&
        entry.resource().kind() == resource_kinds::kRamBytes) {
      LOG(INFO) << "Calling MallocExtension_ReleaseToSystem() with "
                << entry.quantity();
      ::tensorflow::port::MallocExtension_ReleaseToSystem(entry.quantity());
    }
  }
}

template <typename DataType, typename ServableType>
SimpleLoaderSourceAdapter<DataType,
                          ServableType>::~SimpleLoaderSourceAdapter() {}

template <typename DataType, typename ServableType>
typename SimpleLoaderSourceAdapter<DataType, ServableType>::ResourceEstimator
SimpleLoaderSourceAdapter<DataType, ServableType>::EstimateNoResources() {
  return [](const DataType& data, ResourceAllocation* estimate) {
    estimate->Clear();
    return Status::OK();
  };
}

template <typename DataType, typename ServableType>
SimpleLoaderSourceAdapter<DataType, ServableType>::SimpleLoaderSourceAdapter(
    Creator creator, ResourceEstimator resource_estimator)
    : creator_(creator), resource_estimator_(resource_estimator) {}

template <typename DataType, typename ServableType>
Status SimpleLoaderSourceAdapter<DataType, ServableType>::Convert(
    const DataType& data, std::unique_ptr<Loader>* loader) {
  // We copy 'creator_' and 'resource_estimator_', rather than passing via
  // reference, so that the loader we emit is not tied to the adapter, in case
  // the adapter is deleted before the loader.
  const auto creator = creator_;
  const auto resource_estimator = resource_estimator_;
  loader->reset(new SimpleLoader<ServableType>(
      [creator, data](std::unique_ptr<ServableType>* servable) {
        return creator(data, servable);
      },
      [resource_estimator, data](ResourceAllocation* estimate) {
        return resource_estimator(data, estimate);
      }));
  return Status::OK();
}

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_SIMPLE_LOADER_H_
