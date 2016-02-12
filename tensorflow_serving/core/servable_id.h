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

#ifndef TENSORFLOW_SERVING_CORE_SERVABLE_ID_H_
#define TENSORFLOW_SERVING_CORE_SERVABLE_ID_H_

#include <iosfwd>
#include <string>
#include <unordered_map>

#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace serving {

// An identifier for a Servable. Two Servable objects with the
// same identifier are considered semantically equivalent (modulo its loaded-
// ness state).
struct ServableId {
  // The name of the servable stream to which this servable object belongs.
  string name;

  // The sequence number of this servable object in its stream. The primary
  // purpose of 'version' is to uniquely identify a servable object. A
  // secondary purpose is to support inbound requests that identify just a
  // servable stream name but not a specific version; those are routed to the
  // active servable with the largest version number.
  //
  // Must be non-negative.
  int64 version;

  // Returns a string representation of this object. Useful in logging.
  string DebugString() const {
    return strings::StrCat("{name: ", name, " version: ", version, "}");
  }
};

struct HashServableId {
  uint64 operator()(const ServableId& id) const {
    // Hash codes for many common types are remarkably bad, often clustering
    // around the same values of the low and/or high bits for linear
    // sequences of inputs such as 1, 2, 3; or addresses of consecutively
    // allocated objects.  For these cases the default hash function is the
    // identity function on the bit patterns.
    //
    // So we apply a one-to-one mapping to the resulting bit patterns to
    // make the high bits contain more entropy from the entire hash code.
    // It's based on Fibonacci hashing from Knuth's Art of Computer
    // Programming volume 3, section 6.4.
    const uint64 version_hash = [&]() -> uint64 {
      if (id.version >= 0) {
        return std::hash<int64>()(id.version) *
               0x9E3779B9;  // (sqrt(5) - 1)/2 as a binary fraction.
      } else {
        return 0x9E3779B9;
      }
    }();
    // Using version_hash as the seed here to combine the hashes.
    return Hash64(id.name.data(), id.name.size(), version_hash);
  }
};

inline bool operator==(const ServableId& a, const ServableId& b) {
  return a.version == b.version && a.name == b.name;
}

inline bool operator!=(const ServableId& a, const ServableId& b) {
  return !(a == b);
}

inline bool operator<(const ServableId& a, const ServableId& b) {
  const int strcmp_result = a.name.compare(b.name);
  if (strcmp_result != 0) {
    return strcmp_result < 0;
  }
  return a.version < b.version;
}

inline std::ostream& operator<<(std::ostream& out, const ServableId& id) {
  return out << id.DebugString();
}

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_CORE_SERVABLE_ID_H_
