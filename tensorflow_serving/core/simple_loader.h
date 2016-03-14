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

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_serving/core/loader.h"
#include "tensorflow_serving/core/source_adapter.h"
#include "tensorflow_serving/util/any_ptr.h"

namespace tensorflow {
namespace serving {

// SimpleLoader is a wrapper used to create a trivial servable Loader.
// When constructing a SimpleLoader users provide a Creator callback.
// This callback is used in the Load() method to construct a servable of type
// ServableType and populate the servable.
// The servable object is destroyed when Unload() is called.
//
// Example use: create a toy Loader for a servable of type time_t.  Here the
// time is set to the time when Load() is called.
//   std::unique_ptr<Loader> loader(new SimpleLoader<time_t>(
//       {"name_of_servable", kVersion},
//       [](std::unique_ptr<time_t>* servable) {
//           servable->reset(new time_t);
//           *servable = time(nullptr);
//           return Status::OK();
//       }));
//
// IMPORTANT: Use of SimpleLoader abdicates resource safety, i.e. servables
// loaded via SimpleLoaders do not declare their resource usage, and hence the
// serving system cannot enforce resource safety.
template <typename ServableType>
class SimpleLoader : public ResourceUnsafeLoader {
 public:
  // Creator is called in Load and used to create the servable.
  using Creator = std::function<Status(std::unique_ptr<ServableType>*)>;
  explicit SimpleLoader(Creator creator) : creator_(creator) {}
  SimpleLoader() = delete;
  ~SimpleLoader() override = default;

  Status Load() override;

  void Unload() override;

  AnyPtr servable() override { return AnyPtr{servable_.get()}; }

 private:
  Creator creator_;
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
//  - Like SimpleLoader, the emitted loaders' Unload() implementation simply
// calls ServableType's destructor.
//
// For more complex behaviors, SimpleLoaderSourceAdapter is inapplicable. You
// must instead create a SourceAdapter and Loader. That said, you may still be
// able to use one of UnarySourceAdapter or SimpleLoader.
template <typename DataType, typename ServableType>
class SimpleLoaderSourceAdapter
    : public UnarySourceAdapter<DataType, std::unique_ptr<Loader>> {
 public:
  // Creator is called by the produced Loaders' Load() method, and used to
  // create objects of type ServableType. It takes a DataType object as input.
  using Creator =
      std::function<Status(const DataType&, std::unique_ptr<ServableType>*)>;
  explicit SimpleLoaderSourceAdapter(Creator creator) : creator_(creator) {}
  SimpleLoaderSourceAdapter() = delete;
  ~SimpleLoaderSourceAdapter() override = default;

 protected:
  Status Convert(const DataType& path, std::unique_ptr<Loader>* loader) final;

 private:
  Creator creator_;

  TF_DISALLOW_COPY_AND_ASSIGN(SimpleLoaderSourceAdapter);
};

//////////
// Implementation details follow. API users need not read.

template <typename ServableType>
Status SimpleLoader<ServableType>::Load() {
  const Status status = creator_(&servable_);
  return status;
}

template <typename ServableType>
void SimpleLoader<ServableType>::Unload() {
  servable_.reset();
}

template <typename DataType, typename ServableType>
Status SimpleLoaderSourceAdapter<DataType, ServableType>::Convert(
    const DataType& data, std::unique_ptr<Loader>* loader) {
  loader->reset(new SimpleLoader<ServableType>(
      [this, data](std::unique_ptr<ServableType>* servable) {
        return this->creator_(data, servable);
      }));
  return Status::OK();
}

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_SIMPLE_LOADER_H_
