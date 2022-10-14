#include <stdio.h>
#include <cuda.h>
#include <unistd.h>
#include <string.h>

#include <atomic>
#include <string>
#include <iostream>
#include <cstdlib>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <chrono>

#define KEY_NUM 8565
#define MEM_SIZE 1024

#define GB_1 1073741824

using namespace std::string_literals;
namespace bip = boost::interprocess;

static_assert(ATOMIC_INT_LOCK_FREE == 2, "atomic_int must be lock-free");


int main(int argc, char** argv){

	if(argc == 2){	// parent process
		int num_proc = atoi(argv[1]);

		struct shm_remove{
			shm_remove() { bip::shared_memory_object::remove("szMem"); }
			~shm_remove() { bip::shared_memory_object::remove("szMem"); }
		} remover;
		bip::managed_shared_memory segment(bip::create_only, "szMem", MEM_SIZE);

		auto ap = segment.construct<std::atomic_int>("the barrier")(num_proc);
		std::cout<<"The leader counter:"<<*ap<<"\n";

		while(1){
			if(*ap == 0 ){
				break;	
			}
		}
		std::cout<<"Destory barrier\n";

		segment.destroy<std::atomic_int>("the barrier");
	}
	else if(argc==3){

		float *arr, *arr_h;
		float* arr_init;
		cudaMalloc((void**)&arr_init, 4);
		cudaFree(arr_init);

		int shift = atoi(argv[2]);	//argv[1] == num_proc, ignored
		unsigned long long size=4096;
		size<<=shift;
		printf("size : %llu KB",size/1024);

		bip::managed_shared_memory segment1(bip::open_only, "szMem");
		auto bar = segment1.find<std::atomic_int>("the barrier");
		--*bar.first;

		std::cout << "The barrier counter: " << *bar.first << "\n";	
		while(1){
			if(*bar.first == 0)
				break;
		}

		cudaError_t err = cudaMalloc((void**) &arr, size);

		if(err != cudaSuccess){
			std::cout << cudaGetErrorString(err) << "\n";	
		}

		free(arr_h);
		cudaFree(arr);
	}
	else{
		std::cout<<"Bad Usage!\n" \		
				<<"Please run parent process first. ./[bin.exe] [# of processes]\n" \	
				<<"And then run childe process [# of prcoesses] times with shift argument (0: 4KB, 1: 8KB))\n" \
				<<"ex) ./[bin.exe] [# of processes] [shift bit]\n";
	}
	return 0;	
}
