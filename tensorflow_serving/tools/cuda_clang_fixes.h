/* Copyright 2026 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_SERVING_TOOLS_CUDA_CLANG_FIXES_H_
#define TENSORFLOW_SERVING_TOOLS_CUDA_CLANG_FIXES_H_

#if defined(__clang__) && defined(__CUDA__)
#include <vector_types.h>

// Fix for GNU typeof in strict C++ mode when compiling NVIDIA DOCA / NCCL headers with Clang.
#ifndef typeof
#define typeof __typeof__
#endif

// Missing 16-byte uint4 store overload in Clang's __clang_cuda_intrinsics.h for NCCL GIN proxy (__stwt).
__device__ inline void __stwt(uint4* ptr, uint4 value) {
  asm volatile("st.global.wt.v4.u32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(ptr), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w)
               : "memory");
}
#endif

#endif  // TENSORFLOW_SERVING_TOOLS_CUDA_CLANG_FIXES_H_
