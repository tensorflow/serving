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

#ifndef TENSORFLOW_SERVING_CORE_SOURCE_ADAPTER_H_
#define TENSORFLOW_SERVING_CORE_SOURCE_ADAPTER_H_

#include <algorithm>
#include <vector>

#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_serving/core/servable_data.h"
#include "tensorflow_serving/core/source.h"
#include "tensorflow_serving/core/target.h"

namespace tensorflow {
namespace serving {

// An abstraction for a module that receives aspired-version callbacks with data
// of type InputType and converts them into calls with data of type OutputType.
//
// A common example uses InputType=StoragePath, OutputType=unique_ptr<Loader>,
// in which case the module "converts" each incoming storage path into a loader
// capable of loading a (particular type of) servable based on the path.
//
// SourceAdapters are typically stateless. However, as with all Sources they can
// house state that is shared among multiple emitted servables. See the
// discussion in source.h.
template <typename InputType, typename OutputType>
class SourceAdapter : public TargetBase<InputType>, public Source<OutputType> {
 public:
  ~SourceAdapter() = default;

  // This method is implemented in terms of Adapt(), which the implementing
  // subclass must supply.
  void SetAspiredVersions(const StringPiece servable_name,
                          std::vector<ServableData<InputType>> versions) final;

  void SetAspiredVersionsCallback(
      typename Source<OutputType>::AspiredVersionsCallback callback) final;

 private:
  // Given an InputType-based aspired-versions request, produces a corresponding
  // OutputType-based request.
  virtual std::vector<ServableData<OutputType>> Adapt(
      const StringPiece servable_name,
      std::vector<ServableData<InputType>> versions) = 0;

  // The callback for emitting OutputType-based aspired-version lists.
  typename Source<OutputType>::AspiredVersionsCallback outgoing_callback_;

  // Has 'outgoing_callback_' been set yet, so that the SourceAdapter is ready
  // to propagate aspired versions?
  Notification outgoing_callback_set_;
};

// A source adapter that converts InputType instances to OutputType instances
// one at a time (i.e. there is no interaction among members of a given aspired-
// version list). Most source adapters can subclass UnarySourceAdapter, and do
// not need the full generality of SourceAdapter.
//
// Requires OutputType to be default-constructable and updatable in-place.
template <typename InputType, typename OutputType>
class UnarySourceAdapter : public SourceAdapter<InputType, OutputType> {
 public:
  UnarySourceAdapter() = default;
  ~UnarySourceAdapter() override = default;

 private:
  // This method is implemented in terms of Convert(), which the implementing
  // subclass must supply.
  std::vector<ServableData<OutputType>> Adapt(
      const StringPiece servable_name,
      std::vector<ServableData<InputType>> versions) final;

  // Converts a single InputType instance into a corresponding OutputType
  // instance.
  virtual Status Convert(const InputType& data, OutputType* converted_data) = 0;
};

// A source adapter that converts every incoming ServableData<InputType> item
// into an error-containing ServableData<OutputType>. If the incoming data item
// was already an error, the existing error is passed through; otherwise a new
// error Status given via this class's constructor is added.
//
// This class is useful in conjunction with a router, to handle servable data
// items that do not conform to any explicitly-programmed route. Specifically,
// consider a fruit router configured route apples to output port 0, oranges to
// output port 1, and anything else to a final port 2. If we only have proper
// SourceAdapters to handle apples and oranges, we might connect an
// ErrorInjectingSourceAdapter to port 2, to catch any unexpected fruits.
template <typename InputType, typename OutputType>
class ErrorInjectingSourceAdapter
    : public SourceAdapter<InputType, OutputType> {
 public:
  explicit ErrorInjectingSourceAdapter(const Status& error);
  ~ErrorInjectingSourceAdapter() override = default;

 private:
  std::vector<ServableData<OutputType>> Adapt(
      const StringPiece servable_name,
      std::vector<ServableData<InputType>> versions) override;

  // The error status to inject.
  const Status error_;

  TF_DISALLOW_COPY_AND_ASSIGN(ErrorInjectingSourceAdapter);
};

//////////
// Implementation details follow. API users need not read.

template <typename InputType, typename OutputType>
void SourceAdapter<InputType, OutputType>::SetAspiredVersions(
    const StringPiece servable_name,
    std::vector<ServableData<InputType>> versions) {
  outgoing_callback_set_.WaitForNotification();
  outgoing_callback_(servable_name, Adapt(servable_name, std::move(versions)));
}

template <typename InputType, typename OutputType>
void SourceAdapter<InputType, OutputType>::SetAspiredVersionsCallback(
    typename Source<OutputType>::AspiredVersionsCallback callback) {
  outgoing_callback_ = callback;
  outgoing_callback_set_.Notify();
}

template <typename InputType, typename OutputType>
std::vector<ServableData<OutputType>>
UnarySourceAdapter<InputType, OutputType>::Adapt(
    const StringPiece servable_name,
    std::vector<ServableData<InputType>> versions) {
  std::vector<ServableData<OutputType>> adapted_versions;
  for (const ServableData<InputType>& version : versions) {
    if (version.status().ok()) {
      OutputType adapted_data;
      Status adapt_status = Convert(version.DataOrDie(), &adapted_data);
      if (adapt_status.ok()) {
        adapted_versions.emplace_back(
            ServableData<OutputType>{version.id(), std::move(adapted_data)});
      } else {
        adapted_versions.emplace_back(
            ServableData<OutputType>{version.id(), adapt_status});
      }
    } else {
      adapted_versions.emplace_back(
          ServableData<OutputType>{version.id(), version.status()});
    }
  }
  return adapted_versions;
}

template <typename InputType, typename OutputType>
ErrorInjectingSourceAdapter<InputType, OutputType>::ErrorInjectingSourceAdapter(
    const Status& error)
    : error_(error) {
  DCHECK(!error.ok());
}

template <typename InputType, typename OutputType>
std::vector<ServableData<OutputType>>
ErrorInjectingSourceAdapter<InputType, OutputType>::Adapt(
    const StringPiece servable_name,
    std::vector<ServableData<InputType>> versions) {
  std::vector<ServableData<OutputType>> adapted_versions;
  for (const ServableData<InputType>& version : versions) {
    if (version.status().ok()) {
      LOG(INFO) << "Injecting error for servable " << version.id() << ": "
                << error_.error_message();
      adapted_versions.emplace_back(
          ServableData<OutputType>{version.id(), error_});
    } else {
      adapted_versions.emplace_back(
          ServableData<OutputType>{version.id(), version.status()});
    }
  }
  return adapted_versions;
}

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_SOURCE_ADAPTER_H_
