#include <cuda.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char* argv[]){
	cudaFreeArray(0);
	
	float* arr;
	unsigned long long int size=4096; // 0
	cudaError_t err;
	unsigned long long int limit = 34359738368;

	unsigned int arg = atoi(argv[1]);

	size <<= arg;

	printf("size: %llu\n",size);

	err = cudaMalloc((void**)&arr, size);
	if(err != cudaSuccess){
		printf("err: %s\n",cudaGetErrorString(err));
	}


	return 0;
}
