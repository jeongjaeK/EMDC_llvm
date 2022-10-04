#include <cuda.h>
#include <vector>
#include <iostream>

static inline void 
checkDrvError(CUresult res, const char *tok, const char *file, unsigned line)
{
	if( res != CUDA_SUCCESS){
		const char *errStr = NULL;
		(void)cuGetErrorString(res, &errStr);
		std::cerr << file << ":" << line << ' ' << tok << "failed (" << (unsigned)res << "): " << errStr << "\n";
	}
}

#define CHECK_DRV(x) checkDrvError(x, #x, __FILE__, __LINE__);

struct GpuMemAllocDesc{
	void** ptr;
	//	CUdeviceptr* ptr;
	size_t size;
	GpuMemAllocDesc(void** _ptr, size_t _size):ptr(_ptr),size(_size){}	
};

class MergedAllocation{
	size_t sum_size;
	CUmemGenericAllocationHandle hdl;
	CUmemAccessDesc accessDesc;
	CUdeviceptr ptr;

	public:
	MergedAllocation(std::vector<struct GpuMemAllocDesc>& alloc_info, int dev=0){
		sum_size = 0;

		size_t aligned_sz;
		CUmemAllocationProp prop={};
		prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
		prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
		prop.location.id = dev;
		accessDesc.location = prop.location;
		accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

		CHECK_DRV(cuMemGetAllocationGranularity(&aligned_sz, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));		

		for(auto it = alloc_info.begin(); it != alloc_info.end(); ++it){
			sum_size += ((it->size + aligned_sz -1) / aligned_sz) * aligned_sz;
		}


		CHECK_DRV(cuMemCreate(&hdl, sum_size, &prop, 0));
		CHECK_DRV(cuMemAddressReserve(&ptr, sum_size, 0ULL, 0ULL, 0ULL));
		CHECK_DRV(cuMemMap(ptr, sum_size, 0ULL, hdl, 0ULL));
		CHECK_DRV(cuMemSetAccess(ptr, sum_size, &accessDesc, 1ULL));
		std::cout<<"ptr :"<<ptr<<"\n";
		sum_size =0;

		for(auto it = alloc_info.begin(); it != alloc_info.end(); ++it){
			*it->ptr = (void*)(ptr + sum_size);
			sum_size += it->size;
			std::cout<<"it->ptr: "<<*it->ptr<<"\n";
		}

	}
	~MergedAllocation(){
		CHECK_DRV(cuMemUnmap(ptr, sum_size));
		CHECK_DRV(cuMemAddressFree(ptr, sum_size));
		CHECK_DRV(cuMemRelease(hdl));
	}	
};


/* for test */

__global__ void init(float* a, float* b, float* c){
	printf("a: 0x%p, b: 0x%p, c:%p\n",a,b,c);
	printf("a: %f, b: %f, c: %f\n",a[0],b[0],c[0]);
	a[0]=0.0;
	b[0]=0.0;
	c[0]=0.0;
	printf("a: %f, b: %f, c: %f\n",a[0],b[0],c[0]);
}


int main(){
	float *d1, *d2, *d3;

	CUdevice dev;
	int supportsVMM = 0;

	cudaFree(0);

	CHECK_DRV(cuInit(0));

	CHECK_DRV(cuCtxGetDevice(&dev));
	CHECK_DRV(cuDeviceGetAttribute(&supportsVMM, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, dev));

	float *h;
	if(supportsVMM){
		std::vector<GpuMemAllocDesc> alloc_list;
		alloc_list.push_back( GpuMemAllocDesc( (void**)&d1, 1024ULL<<20 ) );
		alloc_list.push_back( GpuMemAllocDesc( (void**)&d2, 1024ULL<<20 ) );
		alloc_list.push_back( GpuMemAllocDesc( (void**)&d3, 1024ULL<<20 ) );

		/*
		   struct GpuMemAllocDesc desc[3];
		   desc[0].ptr = (void**)&d1;
		   desc[0].size = 1024ULL << 20; // 1GB
		   desc[1].ptr = (void**)&d2;
		   desc[1].size = 1024ULL << 20; // 1GB
		   desc[2].ptr = (void**)&d3;
		   desc[2].size = 1024ULL << 20; // 1GB

		   alloc_list.push_back(desc[0]);
		   alloc_list.push_back(desc[1]);
		   alloc_list.push_back(desc[2]);
		 */
		MergedAllocation *ma = new MergedAllocation(alloc_list); 
		std::cout<<"d1: "<<d1<<"\n";
		std::cout<<"d2: "<<d2<<"\n";
		std::cout<<"d3: "<<d3<<"\n";

		h = (float*)malloc(sizeof(float)*4);
		h[0]=1.0; h[1]=1.0; h[2]=1.0;

		cudaMemcpy(d1,&h[0],sizeof(float),cudaMemcpyHostToDevice);
		cudaMemcpy(d2,&h[1],sizeof(float),cudaMemcpyHostToDevice);
		cudaMemcpy(d3,&h[2],sizeof(float),cudaMemcpyHostToDevice);

		init<<<1,1>>>(d1,d2,d3);
		cudaDeviceSynchronize();

		cudaMemcpy(&h[0],d1,sizeof(float),cudaMemcpyDeviceToHost);
		cudaMemcpy(&h[1],d2,sizeof(float),cudaMemcpyDeviceToHost);
		cudaMemcpy(&h[2],d3,sizeof(float),cudaMemcpyDeviceToHost);


		std::cout<<"d1: "<<h[0]<<"\n";
		std::cout<<"d2: "<<h[1]<<"\n";
		std::cout<<"d3: "<<h[2]<<"\n";

		delete(ma);
	}

	cudaMalloc((void**)&d1, 1024ULL<<20);
	cudaMalloc((void**)&d2, 1024ULL<<20);
	cudaMalloc((void**)&d3, 1024ULL<<20);

	init<<<1,1>>>(d1,d2,d3);

	cudaDeviceSynchronize();

	cudaMemcpy(&h[0],d1,sizeof(float),cudaMemcpyDefault);
	cudaMemcpy(&h[1],d2,sizeof(float),cudaMemcpyDefault);
	cudaMemcpy(&h[2],d3,sizeof(float),cudaMemcpyDefault);


	std::cout<<"d1: "<<h[0]<<"\n";
	std::cout<<"d2: "<<h[1]<<"\n";
	std::cout<<"d3: "<<h[2]<<"\n";


	cudaFree(d1);
	return 0;
}
