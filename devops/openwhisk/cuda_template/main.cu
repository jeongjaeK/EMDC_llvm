#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

__global__ void hello(float* input){

	input[threadIdx.x] = 0.0f;
	printf("Hello OW CUDA!");
	
}


int main() {
	float *h_arr, *d_arr;
	size_t alloc_size = sizeof(float)*1024*1024;
	
	h_arr = (float*) malloc(alloc_size);
	cudaMalloc((void**)&d_arr, alloc_size);

	cudaMemcpy(d_arr, h_arr, alloc_size, cudaMemcpyHostToDevice);

	hello<<<1,1>>>(d_arr);
	cudaDeviceSynchronize();

	cudaMemcpy(h_arr,d_arr, alloc_size, cudaMemcpyDeviceToHost);

	cudaFree(d_arr);
	free(h_arr);

	printf("OW CUDA!");
	//std::cout << "{\"msg\": \"The results are correct!\"}";

	return 0;
}
