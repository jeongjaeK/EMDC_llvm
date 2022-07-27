#include <stdio.h>
#include <cuda.h>
#include <unistd.h>
#include <atomic>
#include <string>
#include <iostream>
#include <cstdlib>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <chrono>

#define KEY_NUM 8565
#define MEM_SIZE 1024
#define NUM_PROC 32

#define GB_1 1073741824
#define MB_1 1048576

using namespace std::string_literals;
namespace bip = boost::interprocess;

static_assert(ATOMIC_INT_LOCK_FREE == 2, "atomic_int must be lock-free");


int main(int argc, char** argv){
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
		cudaFreeArray(0);
		cudaDeviceSetLimit(cudaLimitStackSize,1);

		float *arr, *arr_h;

		arr_h = (float*)malloc(1024*1024*1024);

		bip::managed_shared_memory segment1(bip::open_only, "szMem");
		auto bar = segment1.find<std::atomic_int>("the barrier");
		std::cout << "The barrier counter: " << *bar.first << "\n";	
		--*bar.first;
		while(1){
			if(*bar.first == 0)
				break;
		}
		cudaError_t err = cudaMalloc((void**) &arr, GB_1);

		if(err != cudaSuccess){
			std::cout << cudaGetErrorString(err) << "\n";	
		}

		free(arr_h);
		cudaFree(arr);
	}
	return 0;	
}
