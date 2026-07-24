#ifndef CUDA_BF16_FIX_H_
#define CUDA_BF16_FIX_H_

#if defined(__clang__)
#  if defined(__is_identifier)
#    if __is_identifier(__bf16)
#      ifndef __bf16
typedef unsigned short __bf16;
#      endif
#    endif
#  endif
#else
#  ifndef __bf16
typedef unsigned short __bf16;
#  endif
#endif

#endif  // CUDA_BF16_FIX_H_
