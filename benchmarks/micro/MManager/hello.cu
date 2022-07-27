#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

__global__ void hello_kernel(){

	printf("Hello!\n");	
}

int main(){
	cudaFree(0);
	
	hello_kernel<<<1,1>>>();
	system("CUDA_VISIBLE_DEVICES=0 nvidia-smi");

	return 0;
}
