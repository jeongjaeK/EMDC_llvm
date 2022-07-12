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
#define NUM_PROC 4

#define GB_1 1073741824

using namespace std::string_literals;
namespace bip = boost::interprocess;

static_assert(ATOMIC_INT_LOCK_FREE == 2, "atomic_int must be lock-free");


int main(int argc, char** argv){
	cudaFreeArray(0); 	//CUDA context creation

	if(argc == 1){	// init process
		struct shm_remove{
			shm_remove() { bip::shared_memory_object::remove("szMem"); }
			~shm_remove() { bip::shared_memory_object::remove("szMem"); }
		} remover;
		bip::managed_shared_memory segment(bip::create_only, "szMem", MEM_SIZE);	//create barrier(shared memory) among processes

		auto ap = segment.construct<std::atomic_int>("the barrier")(NUM_PROC);		//set barrier as #of processes
		std::cout<<"The leader counter:"<<*ap<<"\n";

		while(1){
			if(*ap == 0 ){
				break;	
			}
		}
		std::cout<<"Destory barrier\n";
		segment.destroy<std::atomic_int>("the barrier");
	}
	else{ 	//execute process

		float *arr, *arr_h;
		arr_h = (float*)malloc(1024*1024*1024);

		/* barrier start */
		bip::managed_shared_memory segment1(bip::open_only, "szMem");
		auto bar = segment1.find<std::atomic_int>("the barrier");
		--*bar.first;
		std::cout << "The barrier counter: " << *bar.first << "\n";	
		while(1){
			if(*bar.first == 0)
				break;
		}
		/* barrier end */
	
		cudaError_t err = cudaMalloc((void**) &arr, GB_1);
		if(err != cudaSuccess){
			std::cout << cudaGetErrorString(err) << "\n";	
		}

		free(arr_h);
		cudaFree(arr);
	}
	return 0;	
}
