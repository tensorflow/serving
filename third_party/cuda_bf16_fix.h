/* Copyright 2026 The TensorFlow Authors. All Rights Reserved. */
#ifndef THIRD_PARTY_CUDA_BF16_FIX_H_
#define THIRD_PARTY_CUDA_BF16_FIX_H_

#if !defined(__clang__) && defined(__GNUC__)
#ifndef __bf16
typedef unsigned short __bf16;
#endif
#endif

#endif  // THIRD_PARTY_CUDA_BF16_FIX_H_
