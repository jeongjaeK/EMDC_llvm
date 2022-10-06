
#include <cuda.h>
#include <vector>
#include <iostream>
#include "cuda_utils.h"
#include <boost/interprocess/managed_shared_memory.hpp>
#include <atomic>

namespace bip = boost::interprocess;

struct GpuMemAllocDesc{
	void** ptr;
	//	CUdeviceptr* ptr;
	size_t size;
	GpuMemAllocDesc(void** _ptr, size_t _size):ptr(_ptr),size(_size){}	
};

class MergedAllocation{
	size_t total_alloc_size;
	CUmemGenericAllocationHandle hdl;
	CUmemAccessDesc accessDesc;
	CUdeviceptr ptr;

	public:
	MergedLockedAllocation(std::vector<struct GpuMemAllocDesc>& alloc_info, int dev=0){
		total_alloc_size = 0;

		size_t aligned_sz;
		CUmemAllocationProp prop={};
		prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
		prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
		prop.location.id = dev;
		accessDesc.location = prop.location;
		accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

		CHECK_DRV(cuMemGetAllocationGranularity(&aligned_sz, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));		

		bip::managed_shared_memory segment(bip::open_only, "MMAPspinlock");
		auto mtx = segment.find<std::atomic_bool>("MMAP mutex");

		for(auto it = alloc_info.begin(); it != alloc_info.end(); ++it){
			total_alloc_size += ((it->size + aligned_sz -1) / aligned_sz) * aligned_sz;
		}

		while(1){ //spinlock
			bool unlocked=false, locked=true;
			//LOCK
			if( (*mtx.first).compare_exchange_strong(unlocked, locked) == true) 
				break;	
		}
		CHECK_DRV(cuMemCreate(&hdl, total_alloc_size, &prop, 0));
		CHECK_DRV(cuMemAddressReserve(&ptr, total_alloc_size, 0ULL, 0ULL, 0ULL));
		CHECK_DRV(cuMemMap(ptr, total_alloc_size, 0ULL, hdl, 0ULL));
		CHECK_DRV(cuMemSetAccess(ptr, total_alloc_size, &accessDesc, 1ULL));

		//UNLOCK
		*mtx.first = false; 
#ifdef DEBUG
		std::cout<<"ptr :"<<ptr<<"\n";
#endif
		total_alloc_size =0;

		for(auto it = alloc_info.begin(); it != alloc_info.end(); ++it){
			*it->ptr = (void*)(ptr + total_alloc_size);
			total_alloc_size += it->size;
#ifdef DEBUG
			std::cout<<"*it->ptr: "<<*it->ptr<<"\n";
#endif 
		}

	}
	~MergedLockedAllocation(){
		CHECK_DRV(cuMemUnmap(ptr, total_alloc_size));
		CHECK_DRV(cuMemAddressFree(ptr, total_alloc_size));
		CHECK_DRV(cuMemRelease(hdl));
	}	
};
