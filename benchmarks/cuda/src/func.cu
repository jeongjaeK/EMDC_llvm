#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>


__global__ void addKernel(float *c, const float *a, const float *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

cudaError_t func1( cudaFuncAttributes* attrib, void (*ptr)() )
{
    return cudaFuncGetAttributes(attrib, ptr);
}

cudaError_t func2( cudaFuncAttributes* attrib, const char* ptr )
{
    return cudaFuncGetAttributes(attrib, ptr);
}


cudaError_t func2( cudaFuncAttributes* attrib, float* ptr )
{
    return func2( attrib, (const char*) ptr);
}

int main()
{
    cudaFuncAttributes attrib;
    cudaError_t err;

    void (*ptr2)() = (void(*)())(addKernel);  // OK on Visual Studio
  //  err = func1(&attrib, ptr2);
  //  printf("result: %s, reg1: %d\n", cudaGetErrorString(err), attrib.numRegs);
    cudaFuncGetAttributes(&attrib, ptr2);
    printf("result: %s, reg1: %d\n", cudaGetErrorString(err), attrib.numRegs);
 
}