#include <cuda.h>
#include <iostream>
#include <nvml.h>
#include <stdio.h>

#include <unistd.h>

using STACK_SIZE_T = size_t;
using DATA_TYPE = float;

#define ALLOC_SIZE 1024
#define MB 1048576
#define KB 1024

__global__ void foo_kernel(){
	printf("Hello\n");	
}

int main(){
	cudaFreeArray(0);
	STACK_SIZE_T stack_size, fifo_size, heap_size, sync_depth, pend_cnt;	

#ifdef NVML
	nvmlDevice_t dev_id_nvml;
	nvmlReturn_t ret_nvml;
	nvmlInit();

	ret_nvml = nvmlDeviceGetHandleByIndex(0,&dev_id_nvml);
	nvmlMemory_t mem_info;
#endif

	DATA_TYPE *arr;
	cudaMalloc((void**)&arr, ALLOC_SIZE);
	cudaFree(arr);

	//system("nvidia-smi");
	cudaDeviceSetLimit(cudaLimitStackSize, 16);
	cudaDeviceGetLimit(&stack_size , cudaLimitStackSize);
	std::cout<<"Limit Stack to size "<<stack_size<<"\n";
#ifdef NVML
	ret_nvml = nvmlDeviceGetMemoryInfo(dev_id_nvml, &mem_info);
	std::cout<<"GMem used: "<<mem_info.used/KB<<"(KB)\n";
	//system("nvidia-smi");
#endif

	cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth,1);
	cudaDeviceGetLimit(&sync_depth, cudaLimitDevRuntimeSyncDepth);
	std::cout<<"Limit sycn depth to "<<sync_depth<<"\n";

	cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 4096);
	cudaDeviceGetLimit(&pend_cnt, cudaLimitDevRuntimePendingLaunchCount);
	std::cout<<"Limit pending count to "<<pend_cnt<<"\n";

	cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 32);
	cudaDeviceGetLimit(&fifo_size, cudaLimitPrintfFifoSize);
	std::cout<<"Limit printf FIFO to size "<<(float)fifo_size/KB<<"\n";
#ifdef NVML	
	ret_nvml = nvmlDeviceGetMemoryInfo(dev_id_nvml, &mem_info);
	std::cout<<"GMem used: "<<mem_info.used/KB<<"(KB)\n";
#endif

	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024);
	cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize);
	std::cout<<"Limit malloc heap to size "<<heap_size/KB<<"\n";
#ifdef NVML
	ret_nvml = nvmlDeviceGetMemoryInfo(dev_id_nvml, &mem_info);
	std::cout<<"GMem used: "<<(float)mem_info.used/KB<<"(KB)\n";
#endif
	foo_kernel<<<1,1>>>();
	cudaDeviceSynchronize();
#ifdef NVML
	nvmlShutdown();
#endif

	system("nvidia-smi");

//	sleep(10);
	return 0;	
}
