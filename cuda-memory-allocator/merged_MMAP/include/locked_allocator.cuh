#include <cuda.h>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <atomic>

#include "cuda_utils.h"

namespace bip = boost::interprocess;

class LOCKEDAllocation{
	size_t sz;
	CUmemGenericAllocationHandle hdl;
	CUMemAccessDessc accessDesc;

public:	
	CUdeviceptr ptr;
	LockedAllocation(size_t size, int dev=0){
		size_t aligned_sz;
		CUmemAllocationProp prop = {};
		prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
		prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
		prop.location.id = dev;
		accessDesc.location = prop.location;
		accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

		CHECK_DRV(cuMemGetAllocationGranularity(&aligned_sz, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
		sz = ((size + aligned_sz - 1) / aligned_sz) * aligned_sz;

		bip::managed_shared_memory segment(bip::open_only, "MMAPspinlock");
		auto mtx = segment.find<std::atomic_bool>("MMAP mutex");

		while(1){
			bool unlocked=false, locked=true;
			if(  (*mtx.first).compare_exchange_strong(unlocked,locked) == true ){ //should be atomic CAS
				break;
			}
		}

		CHECK_DRV(cuMemCreate(&hdl, sz, &prop, 0));
		CHECK_DRV(cuMemAddressReserve(&ptr, sz, 0ULL, 0ULL, 0ULL));
		CHECK_DRV(cuMemMap(ptr, sz, 0ULL, hdl, 0ULL));
		CHECK_DRV(cuMemSetAccess(ptr, sz, &accessDesc, 1ULL));
		*mtx.first = false; //UNLOCK

	}
	~LockedAllocation() {
		CHECK_DRV(cuMemUnmap(ptr, sz));
		CHECK_DRV(cuMemAddressFree(ptr, sz));
		CHECK_DRV(cuMemRelease(hdl));
	}

}


