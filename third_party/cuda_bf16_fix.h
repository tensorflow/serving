/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_SERVING_THIRD_PARTY_CUDA_BF16_FIX_H_
#define TENSORFLOW_SERVING_THIRD_PARTY_CUDA_BF16_FIX_H_

// Workaround for CUDA 12.2 / LLVM 18 / GCC 10 header compatibility issue where
// __bf16 is used in <emmintrin.h> without a definition when compiled in x86
// CUDA mode.
#ifndef __bf16
typedef unsigned short __bf16;
#endif

#endif  // TENSORFLOW_SERVING_THIRD_PARTY_CUDA_BF16_FIX_H_
