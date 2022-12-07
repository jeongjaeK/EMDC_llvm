#include <cuda.h>
#include <iostream>
#include <chrono>
#include <cstring>

namespace chr = std::chrono;

inline void init_arr(float* arr){
	unsigned long long limit = (sizeof arr )/4;
	for(unsigned long long i=0; i<limit; ++i){
		arr[i] = 0.0f;	
	}
}

int main(){
	float *h_arr;
	float *d_arr;
	float *uvm_arr;

	cudaFree(0);

	unsigned long long alloc_size=2*1024*1024; // 2MB	
	for( int i=0; i<=13; ++i, alloc_size <<= 1){
		std::cout<<"alloc size: "<<alloc_size/1024/1024<<"(MB)\n";
		chr::steady_clock::time_point begin = chr::steady_clock::now();
		h_arr = (float*)malloc(alloc_size);
		init_arr(h_arr);
		cudaMalloc((void**)&d_arr, alloc_size);
		cudaMemcpy(d_arr,h_arr,alloc_size,cudaMemcpyDefault);
		cudaMemcpy(h_arr,d_arr,alloc_size,cudaMemcpyDefault);
		init_arr(h_arr);
		cudaFree(d_arr);
		free(h_arr);
		chr::steady_clock::time_point end = chr::steady_clock::now();
		std::cout<<"malloc:\t\t\t"<<chr::duration_cast<chr::microseconds>(end-begin).count()<<"\tus\n";

		begin = chr::steady_clock::now();
		cudaMallocHost((void**)&h_arr,alloc_size);
		init_arr(h_arr);
		cudaMalloc((void**)&d_arr, alloc_size);
		cudaMemcpy(d_arr,h_arr,alloc_size,cudaMemcpyHostToDevice);
		cudaMemcpy(h_arr,d_arr,alloc_size,cudaMemcpyDeviceToHost);
		init_arr(h_arr);
		cudaFree(d_arr);
		cudaFreeHost(h_arr);
		end = chr::steady_clock::now();
		std::cout<<"cudaMallocHost:\t\t"<<chr::duration_cast<chr::microseconds>(end-begin).count()<<"\tus\n";

		begin = chr::steady_clock::now();
		cudaMallocManaged((void**)&uvm_arr, alloc_size);
		init_arr(uvm_arr);
		cudaMemAdvise(uvm_arr, alloc_size, cudaMemAdviseSetPreferredLocation,0);
		cudaMemAdvise(uvm_arr, alloc_size, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);
		cudaMemPrefetchAsync(uvm_arr, alloc_size, 0);
//		cudaMemPrefetchAsync(uvm_arr, alloc_size, cudaCpuDeviceId);
		init_arr(uvm_arr);
		cudaFree(uvm_arr);
		end = chr::steady_clock::now();
		std::cout<<"uvm:\t\t\t"<<chr::duration_cast<chr::microseconds>(end-begin).count()<<"\tus\n";
	}


	
	return 0;	
}
