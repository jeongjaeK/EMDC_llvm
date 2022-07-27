#include <cuda.h>
#include <iostream>

using STACK_SIZE_T = size_t;
using DATA_TYPE = float;

#define ALLOC_SIZE 1024

int main(){
	cudaFreeArray(0);
	
	STACK_SIZE_T stack_size;
	cudaDeviceGetLimit(&stack_size, cudaLimitStackSize);
	std::cout<<"Stack size limit : "<<stack_size<<"\n";

	DATA_TYPE *arr;
	cudaMalloc((void**)&arr, ALLOC_SIZE);
	std::cout<<"After cudaMalloc(default stack size)\n";
	system("nvidia-smi");
	cudaFree(arr);

	std::cout<<"After free...\n";
	system("nvidia-smi");
	cudaDeviceSetLimit(cudaLimitStackSize, stack_size/2);
	cudaDeviceGetLimit(&stack_size, cudaLimitStackSize);
	std::cout<<"Stack size limit : "<<stack_size<<"\n";

	cudaMalloc((void**)&arr, ALLOC_SIZE);
	std::cout<<"After cudaMalloc(stack size/=2)\n";
	system("nvidia-smi");
	cudaFree(arr);


	return 0;	
}
