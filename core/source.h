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

#ifndef TENSORFLOW_SERVING_CORE_SOURCE_H_
#define TENSORFLOW_SERVING_CORE_SOURCE_H_

#include <functional>
#include <vector>

#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow_serving/core/servable_data.h"

namespace tensorflow {
namespace serving {

// An abstraction for a module that sources servables to load, or, more
// precisely, handles to data that can be used to load those servables.
// Examples of such data handles are:
//  - a file-system path to a serialized vocabulary map
//  - a handle to an incoming RPC that specifies a machine-learned model to load
//  - a Loader (see loader.h)
// The data handles are generally assumed to be small.
//
// A Source monitors some external resource (e.g. file system, RPC calls) to
// find out about new servables and/or new versions of servables and/or the
// need to unload servable versions. It uses the provided callback to instruct
// a Target module (e.g. AspiredVersionsManager) which version(s) of a given
// servable to load. Furthermore, depending on the semantics of the Target
// module, the Source implicitly instructs it which ones to unload by omitting
// those servables.
//
// A common case is that a Source emits versions for exactly one servable. An
// even simpler case is that a servable has a single, static version for the
// lifetime of the server.
//
// Sources can house state that is shared among multiple emitted servables, e.g.
//  1. A shared thread pool or other resource that multiple servables use.
//  2. A shared read-only data structure that multiple servables use, to avoid
//     the time and space overhead of replicating the data structure in each
//     servable instance.
// Shared state whose initialization time and size is negligible (e.g. thread
// pools) can be created eagerly by the source, which then embeds a pointer to
// it in each emitted ServableData item. Creation of expensive or large shared
// state should be deferred to the first applicable Loader::Load() call, i.e.
// governed by the manager. Symmetrically, the Loader::Unload() call to the
// final servable using the expensive/large shared state should tear it down.
template <typename T>
class Source {
 public:
  virtual ~Source() = default;

  // A callback for a Source to supply version(s) of a servable to a Target, to
  // be loaded.
  //
  // A single invocation of the callback pertains to a single servable stream
  // (given by 'servable_name'). All versions supplied in a call must be for the
  // servable identified in 'servable_name'. Invocations on different servable
  // streams are orthogonal to one another.
  //
  // Multiple invocations may supply servable-data objects with identical
  // ids (i.e. same servable name and version). Such servable-data objects are
  // treated as semantically equivalent. The recipient will ultimately retain
  // one and discard the rest.
  //
  // If a servable version V is supplied in a first invocation, and subsequently
  // omitted from a second invocation, the implication of omitting V depends on
  // the semantics of the Target of the callback. Certain Targets will interpret
  // V's omission as an implicit instruction to unload V. Each Target must
  // document its semantics in this regard.
  using AspiredVersionsCallback = std::function<void(
      const StringPiece servable_name, std::vector<ServableData<T>> versions)>;

  // Supplies an AspiredVersionsCallback to use. Can be called at most once.
  virtual void SetAspiredVersionsCallback(AspiredVersionsCallback callback) = 0;
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_SOURCE_H_
