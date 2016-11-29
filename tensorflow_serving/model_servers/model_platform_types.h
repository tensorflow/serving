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

#ifndef TENSORFLOW_SERVING_MODEL_SERVERS_PLATFORM_TYPES_H_
#define TENSORFLOW_SERVING_MODEL_SERVERS_PLATFORM_TYPES_H_

// individual model platforms should derive from ModelPlatform
struct ModelPlatform {};

// ModelPlatformTraits should be specialized on a per model-platform basis
template <typename>
struct ModelPlatformTraits {
  static constexpr bool defined = false;
  static const char* name() { return "undefined"; }
  static void GlobalInit(int argc, char** argv) {}
};

// TODO(): merge in to ModelPlatformTraits<TensorFlow>?
constexpr char kTensorFlowModelPlatform[] = "tensorflow";
constexpr char kOtherModelPlatform[] = "other";

// TensorFlow model platform
struct TensorFlow : ModelPlatform {};

// Caffe model platform
struct Caffe : ModelPlatform {};

#endif  // TENSORFLOW_SERVING_MODEL_SERVERS_PLATFORM_TYPES_H_
