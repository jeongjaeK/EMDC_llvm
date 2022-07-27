#include <cuda.h>
#include <stdlib.h>
#include <iostream>
#include <unistd.h>
#include <nvml.h>


#define GB_2 2147483648
#define GB_4 4294967296

int main(){
	cudaFreeArray(0);

	float *arr, *arr_h, *dummy;
	arr_h = (float*)malloc(1024*1024*1024);
	//cudaError_t error = cudaMalloc((void**)&arr, 1024*1024*1024);

	nvmlDevice_t dev_id_nvml;
	nvmlReturn_t ret_nvml;
	nvmlInit();

	ret_nvml = nvmlDeviceGetHandleByIndex(0,&dev_id_nvml);
	nvmlMemory_t mem_info;
	ret_nvml = nvmlDeviceGetMemoryInfo(dev_id_nvml, &mem_info);
	if(ret_nvml == NVML_SUCCESS)
		std::cout<<"before allocation Used gmem: "<<mem_info.used/1024/1024<<"\n";

	cudaError_t error = cudaMalloc((void**)&arr, GB_4);
	if(error != cudaSuccess){
		std::cout<< cudaGetErrorString(error);
	}

	ret_nvml = nvmlDeviceGetMemoryInfo(dev_id_nvml, &mem_info);
	if(ret_nvml == NVML_SUCCESS)
		std::cout<<"after allocation Used gmem: "<<mem_info.used/1024/1024<<"\n";


	cudaMemcpy(arr, arr_h, 1024*1024*1024, cudaMemcpyHostToDevice);
//	cudaMemcpy(arr, arr_h, 512*1024*1024, cudaMemcpyHostToDevice);

	nvmlShutdown();

	free(arr_h);
	cudaFree(arr);
	
	return 0;	
}
