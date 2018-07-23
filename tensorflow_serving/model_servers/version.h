/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_SERVING_MODEL_SERVERS_VERSION_H_
#define TENSORFLOW_SERVING_MODEL_SERVERS_VERSION_H_

// TF Serving Model Server uses semantic versioning, see http://semver.org/.

#define TF_MODELSERVER_MAJOR_VERSION 1
#define TF_MODELSERVER_MINOR_VERSION 10
#define TF_MODELSERVER_PATCH_VERSION 0

// TF_MODELSERVER_VERSION_SUFFIX is non-empty for pre-releases
// (e.g. "-alpha", "-alpha.1", "-beta", "-rc", "-rc.1")
#define TF_MODELSERVER_VERSION_SUFFIX "-dev"

#define TF_MODELSERVER_STR_HELPER(x) #x
#define TF_MODELSERVER_STR(x) TF_MODELSERVER_STR_HELPER(x)

// e.g. "0.5.0" or "0.6.0-alpha".
// clang-format off
#define TF_MODELSERVER_VERSION_STRING \
  (TF_MODELSERVER_STR(TF_MODELSERVER_MAJOR_VERSION) "." TF_MODELSERVER_STR( \
      TF_MODELSERVER_MINOR_VERSION) "." TF_MODELSERVER_STR( \
          TF_MODELSERVER_PATCH_VERSION) TF_MODELSERVER_VERSION_SUFFIX)
// clang-format on

#endif  // TENSORFLOW_SERVING_MODEL_SERVERS_VERSION_H_
