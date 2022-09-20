/* Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <iostream>
#include <atomic>
#include <cuda.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <string>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <cstdlib>

#define MEM_SIZE 1024

using namespace std::string_literals;
namespace bip = boost::interprocess;

static_assert(ATOMIC_INT_LOCK_FREE == 2, "atomic_int must be lock-free");


static inline void
checkRtError(cudaError_t res, const char *tok, const char *file, unsigned line)
{
    if (res != cudaSuccess) {
        std::cerr << file << ':' << line << ' ' << tok
                  << "failed (" << (unsigned)res << "): " << cudaGetErrorString(res) << std::endl;
        abort();
    }
}

#define CHECK_RT(x) checkRtError(x, #x, __FILE__, __LINE__);

static inline void
checkDrvError(CUresult res, const char *tok, const char *file, unsigned line)
{
    if (res != CUDA_SUCCESS) {
        const char *errStr = NULL;
        (void)cuGetErrorString(res, &errStr);
        std::cerr << file << ':' << line << ' ' << tok
                  << "failed (" << (unsigned)res << "): " << errStr << std::endl;
        abort();
    }
}

#define CHECK_DRV(x) checkDrvError(x, #x, __FILE__, __LINE__);

class MMAPAllocation {
    size_t sz;
    CUmemGenericAllocationHandle hdl;
    CUmemAccessDesc accessDesc;
    CUdeviceptr ptr;
public:
    MMAPAllocation(size_t size, int dev = 0) {
        size_t aligned_sz;
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = dev;
        accessDesc.location = prop.location;
        accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

        //CHECK_DRV(cuMemGetAllocationGranularity(&aligned_sz, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        CHECK_DRV(cuMemGetAllocationGranularity(&aligned_sz, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        sz = ((size + aligned_sz - 1) / aligned_sz) * aligned_sz;

        CHECK_DRV(cuMemCreate(&hdl, sz, &prop, 0));
        CHECK_DRV(cuMemAddressReserve(&ptr, sz, 0ULL, 0ULL, 0ULL));
        CHECK_DRV(cuMemMap(ptr, sz, 0ULL, hdl, 0ULL));

		bip::managed_shared_memory segment3(bip::open_only, "szMem2");
		auto bar1 = segment3.find<std::atomic_int>("the barrier2");
		--*bar1.first;
		std::cout << "The barreir count: "<< *bar1.first<<"\n";
		while(1){
			if(*bar1.first ==0)	break;	
		}
	
        CHECK_DRV(cuMemSetAccess(ptr, sz, &accessDesc, 1ULL));
    }
    ~MMAPAllocation() {
        CHECK_DRV(cuMemUnmap(ptr, sz));
        CHECK_DRV(cuMemAddressFree(ptr, sz));
        CHECK_DRV(cuMemRelease(hdl));
    }
};



int main(int argc, char** argv)
{
    //const size_t N = 4ULL;
    int supportsVMM = 0;
    CUdevice dev;

	if(argc == 2){ // parent process
		int num_proc = atoi(argv[1]);

		struct shm_remove{
			shm_remove(){ 
				bip::shared_memory_object::remove("szMem"); 
				bip::shared_memory_object::remove("szMem2");
				bip::shared_memory_object::remove("szMem3");
			}	
			~shm_remove(){
				 bip::shared_memory_object::remove("szMem"); 
				 bip::shared_memory_object::remove("szMem2");
				 bip::shared_memory_object::remove("szMem3");
			}	
		}remover;

		bip::managed_shared_memory segment(bip::create_only, "szMem", MEM_SIZE);
		auto ap = segment.construct<std::atomic_int>("the barrier")(num_proc);

		bip::managed_shared_memory segment2(bip::create_only, "szMem2", MEM_SIZE);
		auto ap2 = segment2.construct<std::atomic_int>("the barrier2")(num_proc);

		bip::managed_shared_memory segment3(bip::create_only, "szMem3", MEM_SIZE);
		auto ap3 = segment3.construct<std::atomic_int>("the barrier3")(num_proc);

		std::cout << "The leader counter : "<<*ap<<"\n";
		while(1){
			if(*ap == 0){
				break;
			}	
		}
		std::cout << "cuMemCreate Destroy barrier\n";

		std::cout << "The leader counter : "<<*ap2<<"\n";
		while(1){
			if(*ap2 == 0){
				break;
			}	
		}
		std::cout << "cuMemSetAccess Destroy barrier\n";

		std::cout << "The leader counter : "<<*ap2<<"\n";
		while(1){
			if(*ap3 == 0){
				break;
			}	
		}
		std::cout << "cuMemRelease Destroy barrier\n";

		segment.destroy<std::atomic_int>("the barrier");
		segment2.destroy<std::atomic_int>("the barrier2");
		segment3.destroy<std::atomic_int>("the barrier3");

	}
	else if(argc == 3){

		size_t ALLOC_SIZE = 4096ULL << atoi(argv[2]);

		std::cout<<"Allocation size : "<<ALLOC_SIZE<<"\n";

		CHECK_RT(cudaFree(0));  // Force and check the initialization of the runtime

		CHECK_DRV(cuCtxGetDevice(&dev));
		CHECK_DRV(cuDeviceGetAttribute(&supportsVMM, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, dev));
		int *x = nullptr;

		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

		CHECK_RT(cudaMalloc(&x, 4096));
		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

		std::cout<<"cudaMalloc elapsed \t\t"<<std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count()<<"us \n";

		CHECK_RT(cudaFree(x));

		/* decrease barrier by 1 */
		bip::managed_shared_memory segment1(bip::open_only, "szMem");
		auto bar = segment1.find<std::atomic_int>("the barrier");
		--*bar.first;
		std::cout << "The barreir count: "<< *bar.first<<"\n";
		while(1){
			if(*bar.first ==0)	break;	
		}
		/* all threads are synchronized  */

		if (supportsVMM) {
			// Now use the Virtual Memory Management APIs
			begin = std::chrono::steady_clock::now();
			MMAPAllocation *allocMMAP = new MMAPAllocation(ALLOC_SIZE);
			end = std::chrono::steady_clock::now();
			std::cout<<"MMAP Allocation elapsed \t"<<std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count()<<"us \n";

			bip::managed_shared_memory segment3(bip::open_only, "szMem3");
			auto bar3 = segment3.find<std::atomic_int>("the barrier3");
			--*bar3.first;
			std::cout << "The barreir count: "<< *bar3.first<<"\n";
			while(1){
				if(*bar3.first ==0)	break;	
			}
			delete(allocMMAP);
		}

		/*
		   CUdeviceptr ptr;
		   begin = std::chrono::steady_clock::now();
		CHECK_DRV(cuMemAlloc(&ptr, ALLOC_SIZE));
		end = std::chrono::steady_clock::now();
		std::cout<<"cuMemAlloc elapsed \t"<<std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count()<<"us \n";
		CHECK_DRV(cuMemFree(ptr));
		*/
	}
	else{
		;	
	}
	return 0;
}
