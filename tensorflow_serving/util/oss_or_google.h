/* Copyright 2019 Google Inc. All Rights Reserved.

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
#ifndef TENSORFLOW_SERVING_UTIL_OSS_OR_GOOGLE_H_
#define TENSORFLOW_SERVING_UTIL_OSS_OR_GOOGLE_H_

#define TENSORFLOW_SERVING_OSS

namespace tensorflow {
namespace serving {

// Used to distinguish the context of the code; whether it's part of our OSS
// distribution or within Google.
//
// This is useful in cases where we want to enable/disable running some piece of
// code based on whether we are in/out of OSS.
//
// NB that the method is marked 'constexpr' so that the value can be used as
// a compile-time constant.
inline constexpr bool IsTensorflowServingOSS() {
#ifdef TENSORFLOW_SERVING_GOOGLE
  return false;
#else
  return true;
#endif
}

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_UTIL_OSS_OR_GOOGLE_H_
