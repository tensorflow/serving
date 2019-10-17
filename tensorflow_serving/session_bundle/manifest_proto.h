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

#ifndef TENSORFLOW_SERVING_SESSION_BUNDLE_MANIFEST_PROTO_H_
#define TENSORFLOW_SERVING_SESSION_BUNDLE_MANIFEST_PROTO_H_

#include "tensorflow_serving/util/oss_or_google.h"

#ifdef TENSORFLOW_SERVING_GOOGLE
#include "tensorflow_serving/session_bundle/google/manifest.pb.h"
#else
#include "tensorflow_serving/session_bundle/oss/manifest.pb.h"
#endif

#endif  // TENSORFLOW_SERVING_SESSION_BUNDLE_MANIFEST_PROTO_H_
