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

#include "absl/types/optional.h"
#include "absl/types/variant.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_serving/core/loader.h"
#include "tensorflow_serving/core/source_adapter.h"
#include "tensorflow_serving/resources/resource_util.h"
#include "tensorflow_serving/resources/resource_values.h"
#include "tensorflow_serving/util/any_ptr.h"

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
//       return Status();
//   };
//   auto resource_estimator = [](ResourceAllocation* estimate) {
//       estimate->mutable_...(...)->set_...(...);
//       return Status();
//   };
//   std::unique_ptr<Loader> loader(new SimpleLoader<time_t>(
//       servable_creator, resource_estimator));
//
// This class is not thread-safe. Synchronization is assumed to be done by the
// caller.
template <typename ServableType>
class SimpleLoader : public Loader {
 public:
  // Creator is called in Load and used to create the servable.
  using Creator = std::function<Status(std::unique_ptr<ServableType>*)>;
  using CreatorWithMetadata =
      std::function<Status(const Metadata&, std::unique_ptr<ServableType>*)>;
  using CreatorVariant = absl::variant<Creator, CreatorWithMetadata>;

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

  // Constructor that takes a single resource estimator, to use for estimating
  // the resources needed during load as well as post-load.
  SimpleLoader(Creator creator, ResourceEstimator resource_estimator);

  // Similar to the above constructor, but accepts a CreatorWithMetadata
  // function.
  SimpleLoader(CreatorWithMetadata creator_with_metadata,
               ResourceEstimator resource_estimator);

  // Constructor that takes two resource estimators: one to use for estimating
  // the resources needed during load, as well as a second one that gives a
  // different estimate after loading has finished. See the documentation on
  // Loader::EstimateResources() for (a) potential reasons the estimate might
  // decrease, and (b) correctness constraints on how the estimate is allowed to
  // change over time.
  SimpleLoader(Creator creator, ResourceEstimator resource_estimator,
               ResourceEstimator post_load_resource_estimator);

  // Similar to the above constructor, but accepts a CreatorWithMetadata
  // function.
  SimpleLoader(CreatorWithMetadata creator_with_metadata,
               ResourceEstimator resource_estimator,
               ResourceEstimator post_load_resource_estimator);

  // Constructor which accepts all variations of the params.
  SimpleLoader(CreatorVariant creator_variant,
               ResourceEstimator resource_estimator,
               absl::optional<ResourceEstimator> post_load_resource_estimator);

  ~SimpleLoader() override = default;

  Status EstimateResources(ResourceAllocation* estimate) const override;

  // REQUIRES: That the ctor with Creator be used, otherwise returns an error
  // status.
  Status Load() override;

  Status LoadWithMetadata(const Metadata& metadata) override;

  void Unload() override;

  AnyPtr servable() override { return AnyPtr{servable_.get()}; }

 private:
  Status EstimateResourcesPostLoad();

  CreatorVariant creator_variant_;

  // A function that estimates the resources needed to load the servable.
  ResourceEstimator resource_estimator_;

  // An optional function that estimates the resources needed for the servable
  // after it has been loaded. (If omitted, 'resource_estimator_' should be used
  // for all estimates, i.e. before, during and after load.)
  absl::optional<ResourceEstimator> post_load_resource_estimator_;

  // The memoized estimated resource requirement of the servable.
  mutable absl::optional<ResourceAllocation> memoized_resource_estimate_
      TF_GUARDED_BY(memoized_resource_estimate_mu_);
  mutable mutex memoized_resource_estimate_mu_;

  std::unique_ptr<ResourceUtil> resource_util_;
  Resource ram_resource_;

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
    return Status();
  };
}

template <typename ServableType>
SimpleLoader<ServableType>::SimpleLoader(Creator creator,
                                         ResourceEstimator resource_estimator)
    : SimpleLoader(CreatorVariant(creator), resource_estimator, absl::nullopt) {
}

template <typename ServableType>
SimpleLoader<ServableType>::SimpleLoader(
    CreatorWithMetadata creator_with_metadata,
    ResourceEstimator resource_estimator)
    : SimpleLoader(CreatorVariant(creator_with_metadata), resource_estimator,
                   absl::nullopt) {}

template <typename ServableType>
SimpleLoader<ServableType>::SimpleLoader(
    Creator creator, ResourceEstimator resource_estimator,
    ResourceEstimator post_load_resource_estimator)
    : SimpleLoader(CreatorVariant(creator), resource_estimator,
                   {post_load_resource_estimator}) {}

template <typename ServableType>
SimpleLoader<ServableType>::SimpleLoader(
    CreatorWithMetadata creator_with_metadata,
    ResourceEstimator resource_estimator,
    ResourceEstimator post_load_resource_estimator)
    : SimpleLoader(CreatorVariant(creator_with_metadata), resource_estimator,
                   {post_load_resource_estimator}) {}

template <typename ServableType>
SimpleLoader<ServableType>::SimpleLoader(
    CreatorVariant creator_variant, ResourceEstimator resource_estimator,
    absl::optional<ResourceEstimator> post_load_resource_estimator)
    : creator_variant_(creator_variant),
      resource_estimator_(resource_estimator),
      post_load_resource_estimator_(post_load_resource_estimator) {
  ResourceUtil::Options resource_util_options;
  resource_util_options.devices = {{device_types::kMain, 1}};
  resource_util_ =
      std::unique_ptr<ResourceUtil>(new ResourceUtil(resource_util_options));

  ram_resource_ = resource_util_->CreateBoundResource(
      device_types::kMain, resource_kinds::kRamBytes);
}

template <typename ServableType>
Status SimpleLoader<ServableType>::EstimateResources(
    ResourceAllocation* estimate) const {
  mutex_lock l(memoized_resource_estimate_mu_);
  if (memoized_resource_estimate_) {
    *estimate = *memoized_resource_estimate_;
    return Status();
  }

  // Compute and memoize the resource estimate.
  TF_RETURN_IF_ERROR(resource_estimator_(estimate));
  memoized_resource_estimate_ = *estimate;
  return Status();
}

template <typename ServableType>
Status SimpleLoader<ServableType>::Load() {
  if (absl::holds_alternative<CreatorWithMetadata>(creator_variant_)) {
    return errors::FailedPrecondition(
        "SimpleLoader::Load() called even though "
        "SimpleLoader::CreatorWithMetadata was setup. Please use "
        "SimpleLoader::LoadWithMetadata() instead.");
  }
  TF_RETURN_IF_ERROR(absl::get<Creator>(creator_variant_)(&servable_));
  return EstimateResourcesPostLoad();
}

template <typename ServableType>
Status SimpleLoader<ServableType>::LoadWithMetadata(const Metadata& metadata) {
  if (absl::holds_alternative<CreatorWithMetadata>(creator_variant_)) {
    TF_RETURN_IF_ERROR(
        absl::get<CreatorWithMetadata>(creator_variant_)(metadata, &servable_));
  } else {
    TF_RETURN_IF_ERROR(absl::get<Creator>(creator_variant_)(&servable_));
  }
  return EstimateResourcesPostLoad();
}

template <typename ServableType>
Status SimpleLoader<ServableType>::EstimateResourcesPostLoad() {
  if (post_load_resource_estimator_) {
    // Save the during-load estimate (may be able to use the memoized value).
    ResourceAllocation during_load_resource_estimate;
    TF_RETURN_IF_ERROR(EstimateResources(&during_load_resource_estimate));

    // Obtain the post-load estimate, and store it as the memoized value.
    ResourceAllocation post_load_resource_estimate;
    TF_RETURN_IF_ERROR(
        (*post_load_resource_estimator_)(&post_load_resource_estimate));
    {
      mutex_lock l(memoized_resource_estimate_mu_);
      memoized_resource_estimate_ = post_load_resource_estimate;
    }

    // Release any transient memory used only during load to the OS.
    const uint64_t during_load_ram_estimate = resource_util_->GetQuantity(
        ram_resource_, during_load_resource_estimate);
    const uint64_t post_load_ram_estimate =
        resource_util_->GetQuantity(ram_resource_, post_load_resource_estimate);
    if (post_load_ram_estimate < during_load_ram_estimate) {
      const uint64_t transient_ram_estimate =
          during_load_ram_estimate - post_load_ram_estimate;
      LOG(INFO) << "Calling MallocExtension_ReleaseToSystem() after servable "
                   "load with "
                << transient_ram_estimate;
      ::tensorflow::port::MallocExtension_ReleaseToSystem(
          transient_ram_estimate);
    }
  }

  return Status();
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
  const uint64_t memory_estimate =
      resource_util_->GetQuantity(ram_resource_, resource_estimate);
  if (memory_estimate > 0) {
    LOG(INFO) << "Calling MallocExtension_ReleaseToSystem() after servable "
                 "unload with "
              << memory_estimate;
    ::tensorflow::port::MallocExtension_ReleaseToSystem(memory_estimate);
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
    return Status();
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
  return Status();
}

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_SIMPLE_LOADER_H_
