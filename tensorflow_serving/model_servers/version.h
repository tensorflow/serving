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

// TF Serving Model Server uses semantic versioning for releases
// - see http://semver.org/. For nightlies, a git hash is used to track the
// build via linkstamping
#define TF_MODELSERVER_STR_HELPER(x) #x
#define TF_MODELSERVER_STR(x) TF_MODELSERVER_STR_HELPER(x)

#define TF_MODELSERVER_MAJOR_VERSION 1
#define TF_MODELSERVER_MINOR_VERSION 15
#define TF_MODELSERVER_PATCH_VERSION 0
// TF_MODELSERVER_VERSION_SUFFIX is non-empty for pre-releases
// (e.g. "-alpha", "-alpha.1", "-beta", "-rc", "-rc.1")
#define TF_MODELSERVER_VERSION_SUFFIX "-rc2"

#ifndef TF_MODELSERVER_VERSION_NO_META
// TF_MODELSERVER_BUILD_TAG can be set to be nightly for nightly builds
#ifndef TF_MODELSERVER_BUILD_TAG
#define TF_MODELSERVER_META_TAG "+dev"
#else
#define TF_MODELSERVER_META_TAG "+" TF_MODELSERVER_STR(TF_MODELSERVER_BUILD_TAG)
#endif
#define TF_MODELSERVER_META_SCM_HASH ".sha." BUILD_SCM_REVISION
#else
#define TF_MODELSERVER_META_TAG ""
#define TF_MODELSERVER_META_SCM_HASH ""
#endif

// e.g. "0.5.0+nightly.sha.a1b2c3d" or "0.6.0-rc1".
// clang-format off
#define TF_MODELSERVER_VERSION_STRING \
  (TF_MODELSERVER_STR(TF_MODELSERVER_MAJOR_VERSION) "." TF_MODELSERVER_STR( \
    TF_MODELSERVER_MINOR_VERSION) "." TF_MODELSERVER_STR( \
      TF_MODELSERVER_PATCH_VERSION) TF_MODELSERVER_VERSION_SUFFIX \
        TF_MODELSERVER_META_TAG TF_MODELSERVER_META_SCM_HASH)
// clang-format on

extern "C" {
const char* TF_Serving_Version();
}

#endif  // TENSORFLOW_SERVING_MODEL_SERVERS_VERSION_H_
