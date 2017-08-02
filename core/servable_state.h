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

#ifndef TENSORFLOW_SERVING_CORE_SERVABLE_STATE_H_
#define TENSORFLOW_SERVING_CORE_SERVABLE_STATE_H_

#include <string>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_serving/core/servable_id.h"

namespace tensorflow {
namespace serving {

// The state of a servable. Typically published on an EventBus by a manager.
// Since managers (and EventBus implementations) can in general have multiple
// threads, this is really a snapshot of a servable's recent state, not a
// guarantee about its current state.
//
// Note that this is a semantic state, meant to be independent of the
// implementation of a particular kind of manager.
struct ServableState {
  // The identifier of the servable whose state is represented.
  ServableId id;

  // The state of the servable as maintained by a manager. These states can only
  // transition from higher to lower ones on this list.
  enum class ManagerState : int {
    // The manager is tracking this servable, but has not initiated any action
    // pertaining to it.
    kStart,

    // The manager has decided to load this servable. In particular, checks
    // around resource availability and other aspects have passed, and the
    // manager is about to invoke the loader's Load() method.
    kLoading,

    // The manager has successfully loaded this servable and made it available
    // for serving (i.e. GetServableHandle(id) will succeed). To avoid races,
    // this state is not reported until *after* the servable is made available.
    kAvailable,

    // The manager has decided to make this servable unavailable, and unload it.
    // To avoid races, this state is reported *before* the servable is made
    // unavailable.
    kUnloading,

    // This servable has reached the end of its journey in the manager. Either
    // it loaded and ultimately unloaded successfully, or it hit an error at
    // some point in its lifecycle.
    kEnd,
  };
  ManagerState manager_state;

  // Whether anything has gone wrong with this servable. If not OK, the error
  // could be something that occurred in a Source or SourceAdapter, in the
  // servable's Loader, in the Manager, or elsewhere. All errors pertaining to
  // the servable are reported here, regardless of origin.
  Status health;

  // Returns a string representation of this object. Useful in logging.
  string DebugString() const {
    return strings::StrCat("id: ", id.DebugString(), " manager_state: ",
                           static_cast<int>(manager_state), " health: ",
                           health.ToString());
  }
};

inline bool operator==(const ServableState& a, const ServableState& b) {
  return a.id == b.id && a.manager_state == b.manager_state &&
         a.health == b.health;
}

inline bool operator!=(const ServableState& a, const ServableState& b) {
  return !(a == b);
}

inline std::ostream& operator<<(std::ostream& os, const ServableState& state) {
  return os << state.DebugString();
}

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_SERVABLE_STATE_H_
