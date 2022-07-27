#include <stdio.h>
#include <cuda.h>
#include <unistd.h>
//#include <sys/shm.h>
//#include <sys/ipc.h>
//#include <stdatomic.h>

#include <atomic>
#include <string>
#include <iostream>
#include <cstdlib>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <chrono>

#define KEY_NUM 8565
#define MEM_SIZE 1024
#define NUM_PROC 4

#define GB_1 1073741824

using namespace std::string_literals;
namespace bip = boost::interprocess;

static_assert(ATOMIC_INT_LOCK_FREE == 2, "atomic_int must be lock-free");


int main(int argc, char** argv){
	cudaFreeArray(0);

	if(argc == 1){	// parent process
		struct shm_remove{
			shm_remove() { bip::shared_memory_object::remove("szMem"); }
			~shm_remove() { bip::shared_memory_object::remove("szMem"); }
		} remover;
		bip::managed_shared_memory segment(bip::create_only, "szMem", MEM_SIZE);

		auto ap = segment.construct<std::atomic_int>("the barrier")(NUM_PROC);
		std::cout<<"The leader counter:"<<*ap<<"\n";

		while(1){
			if(*ap == 0 ){
				break;	
			}
		}
		std::cout<<"Destory barrier\n";

		segment.destroy<std::atomic_int>("the barrier");
	}
	else{

		float *arr, *arr_h;
		float *dummy;

		cudaStream_t s1;
		cudaStreamCreate(&s1);

		cudaMalloc((void**)&dummy, 4);

		arr_h = (float*)malloc(1024*1024*1024);
//		cudaMalloc((void**) &arr, 1024*1024*1024);

		cudaStreamSynchronize(s1);

		bip::managed_shared_memory segment1(bip::open_only, "szMem");
		auto bar = segment1.find<std::atomic_int>("the barrier");
		--*bar.first;

		std::cout << "The barrier counter: " << *bar.first << "\n";	
		while(1){
			if(*bar.first == 0)
				break;
		}

//		cudaMemcpy(arr, arr_h, 1024*1024*1024, cudaMemcpyHostToDevice);

//		cudaError_t err = cudaMalloc((void**) &arr, 1024*1024*1024);
		//cudaError_t err = cudaMalloc((void**) &arr, GB_1);
		cudaError_t err = cudaMallocAsync((void**) &arr, GB_1, s1);

		if(err != cudaSuccess){
			std::cout << cudaGetErrorString(err) << "\n";	
		}

		free(arr_h);
		cudaFree(arr);
	}
	return 0;	
}
