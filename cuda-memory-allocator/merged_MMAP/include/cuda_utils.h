#ifndef __CUDA_UTILS_H__
#define __CUDA_UTILS_H__

#include "cuda_utils.cuh"

static inline void checkDrvError(CUresult, const char, const char, unsigned line);

#define CHECK_DRV(x) checkDrvError(x, #x, __FILE__, __LINE__);

#endif
