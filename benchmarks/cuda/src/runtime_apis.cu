#include <cuda.h>
#include <stdlib.h>
#include <string>
#include <iostream>

using namespace std;

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

#ifdef DEBUG
inline void print_k(string msg){
	string cmd = "echo 'HSI:"+ msg +"'| sudo tee /dev/kmsg";
	system(cmd.c_str());
}
#else
	#define print_k(x) ;
#endif

__global__ void foo(float* in, float* in2, float* output){


}

int main(){
	int device_id, attr_val, num_devices;
	cudaDeviceProp prop;
	float* arr;
	cudaFuncCache cacheConfig;
	cudaMemPool_t memPool;
	size_t limitValue;
	int P2PAttr;
	cudaSharedMemConfig SHMEMConfig;
	int leastPri, greatestPri;
	unsigned int device_flags;
	//size_t maxWidthInElements;
	//cudaChannelFormatDesc cfmt={1,1,1,1};
	//void nvSciSyncAttrList;
	
	/* init CUDA runtime */
	print_k("Init CUDA");
	cudaFree(0);
	print_k("Init CUDA done!");
	
	print_k("Analysis begin");
	
	/* start of Device Management */
	print_k("cudaGetDeviceProperties");
	CHECK_RT(	cudaGetDeviceProperties(&prop, 0)		);
	print_k("cudaChooseDevice");
	CHECK_RT(	cudaChooseDevice(&device_id, &prop)		);	
	print_k("cudaDeviceGetAttribute");
	CHECK_RT(	cudaDeviceGetAttribute(&attr_val, cudaDevAttrMemoryPoolsSupported,0)	);
	print_k("cudaDeviceGetCacheConfig");
	CHECK_RT(	cudaDeviceGetCacheConfig(&cacheConfig)	);
	print_k("cudaDeviceGetDefaultMemPool");
	CHECK_RT(	cudaDeviceGetDefaultMemPool(&memPool, 0)	);
	print_k("cudaDeviceGetLimit");
	CHECK_RT(	cudaDeviceGetLimit(&limitValue, cudaLimitStackSize)	);
	//print_k("cudaDeviceGetNvSciSyncAttributes");
	//CHECK_RT(	cudaDeviceGetNvSciSyncAttributes(&nvSciSyncAttrList, 0, cudaNvSciSyncAttrSignal) 	);
	print_k("cudaDeviceGetP2PAttribute");
	CHECK_RT(	cudaDeviceGetP2PAttribute(&P2PAttr, cudaDevP2PAttrAccessSupported, 0, 1)		);
	//cudaDeviceGetPCIBusId 
	print_k("cudaDeviceGetSharedMemConfig");
	CHECK_RT(	cudaDeviceGetSharedMemConfig(&SHMEMConfig)	);
	print_k("cudaDeviceGetStreamPriorityRange");
	CHECK_RT(	cudaDeviceGetStreamPriorityRange (&leastPri, &greatestPri)		);
	//print_k("cudaDeviceGetTexture1DLinearMaxWidth");
	//CHECK_RT(	cudaDeviceGetTexture1DLinearMaxWidth(&maxWidthInElements, &cfmt,0)	);
	print_k("cudaDeviceReset");
	CHECK_RT(	cudaDeviceReset()	);
	print_k("cudaDeviceSetCacheConfig");
	CHECK_RT(	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared)	);	//cudaFuncCachePreferNone: no preference for shared memory or L1 (default); cudaFuncCachePreferShared: prefer larger shared memory and smaller L1 cache;cudaFuncCachePreferL1: prefer larger L1 cache and smaller shared memory;cudaFuncCachePreferEqual: prefer equal size L1 cache and shared memory
	print_k("cudaDeviceSetLimit");
	CHECK_RT(	cudaDeviceSetLimit(cudaLimitStackSize,512) 		);
	print_k("cudaDeviceSetSharedMemConfig");
	CHECK_RT(	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte) 	);
	print_k("cudaGetDevice");
	CHECK_RT(	cudaGetDevice(&device_id)	);
	print_k("cudaGetDeviceCount");
	CHECK_RT(	cudaGetDeviceCount(&num_devices)	);
	print_k("cudaGetDeviceFlags");
	CHECK_RT(	cudaGetDeviceFlags(&device_flags)	);

	print_k("cudaSetDevice");
	CHECK_RT(	cudaSetDevice(0) 	);
	print_k("cudaSetDeviceFlags");
	CHECK_RT(	cudaSetDeviceFlags(cudaDeviceScheduleAuto) 		);
	/* end of Device Management*/

	/* start of Error Handling */
	cudaError_t lastError;
	print_k("cudaGetLastError");	
	CHECK_RT(	lastError = cudaGetLastError()		);
	print_k("cudaPeekAtLastError");
	CHECK_RT(	lastError = cudaPeekAtLastError()	);
	/* end of Error Handling */

	/* start of Stream Management */
	cudaStream_t hStream, hStreamFlags, hStreamPri;
	cudaStreamAttrValue streamAttrVal;
	unsigned int streamFlags;
	int streamPri;

	print_k("cudaCtxResetPersistingL2Cache");
	CHECK_RT(	cudaCtxResetPersistingL2Cache()		);
	print_k("cudaStreamCreate");
	CHECK_RT(	cudaStreamCreate(&hStream)		);
	print_k("cudaStreamCreateWithFlags");
	CHECK_RT(	cudaStreamCreateWithFlags(&hStreamFlags, cudaStreamNonBlocking)		);
	print_k("cudaStreamCreateWithPriority");
	CHECK_RT(	cudaStreamCreateWithPriority(&hStreamPri, cudaStreamDefault, 1)		);
	print_k("cudaStreamCopyAttributes"); // src, dst
	CHECK_RT(	cudaStreamCopyAttributes(hStream, hStreamFlags)		);
	print_k("cudaStreamDestroy");
	CHECK_RT(	cudaStreamDestroy(hStream)		);
	//print_k("cudaStreamGetAttribute");
	//CHECK_RT(	cudaStreamGetAttribute(hStreamPri, cudaAccessPolicyWindow.num_bytes, &streamAttrVal)		);
	print_k("cudaStreamGetFlags");
	CHECK_RT(	cudaStreamGetFlags(hStreamPri, &streamFlags)		);
	print_k("cudaStreamGetPriority");
	CHECK_RT(	cudaStreamGetPriority(hStreamFlags, &streamPri)		);
	print_k("cudaStreamQuery");
	CHECK_RT(	cudaStreamQuery(hStreamPri)		);
	print_k("cudaStreamSynchronize");
	CHECK_RT(	cudaStreamSynchronize(hStreamPri)		);
	/* end of Stream Management */

	/* start of Event Management */
	cudaEvent_t event, eventFlags;

	print_k("cudaEventCreate");
	CHECK_RT(	cudaEventCreate(&event)		);
	print_k("cudaEventCreateWithFlags");
	CHECK_RT(	cudaEventCreateWithFlags(&eventFlags, cudaEventDefault)		);
	print_k("cudaEventDestroy");
	CHECK_RT(	cudaEventDestroy(eventFlags)		);
	print_k("cudaEventRecord");
	CHECK_RT(	cudaEventRecord(event)		);
	print_k("cudaEventQuery");
	CHECK_RT(	cudaEventQuery(event)		);
	print_k("cudaEventSynchronize");
	CHECK_RT(	cudaEventSynchronize(event)		);
	//print_k("cudaEventElapsedTime");
	//CHECK_RT(	cudaEventElapsedTime()	);
	/* end of Event Management */

	/* start of External REsource Interoperability */
	/* end of External REsource Interoperability */

	/* start of Execution Control */
	cudaFuncAttributes funcAttr;
	void (*p_foo)() = (void(*)())(foo);
	print_k("cudaFuncGetAttributes");
	CHECK_RT(	cudaFuncGetAttributes(&funcAttr, p_foo )		);
	print_k("cudaFuncSetAttribute\tSharedMemoryCarveout");
	CHECK_RT(	cudaFuncSetAttribute(p_foo, cudaFuncAttributePreferredSharedMemoryCarveout, 1)		);
	print_k("cudaFuncSetAttribute\tMaxDynamicSharedMemorySize");
	CHECK_RT(	cudaFuncSetAttribute(p_foo, cudaFuncAttributeMaxDynamicSharedMemorySize, 1024*64)		);	
	print_k("cudaFuncSetCacheConfig\tPreferL1");
	CHECK_RT(	cudaFuncSetCacheConfig(p_foo, cudaFuncCachePreferL1)		);
	print_k("cudaFuncSetSharedMemConfig");
	CHECK_RT(	cudaFuncSetSharedMemConfig(p_foo, cudaSharedMemBankSizeDefault)		);
	//print_k("cudaLaunchCooperativeKernel");	
	//CHECK_RT(	cudaLaunchCooperativeKernel(p_foo, (1,1,1), (1,1,1), (a,b,c), )		);
	/* end of Execution Control */

	/* start of Occupancy */
	size_t dynamicSmemSize;
	int numBlocks, numClusters;
	cudaLaunchConfig_t launchConfig;

	print_k("cudaOccupancyAvailableDynamicSMemPerBlock");
	CHECK_RT(	cudaOccupancyAvailableDynamicSMemPerBlock(&dynamicSmemSize,foo,1,1)		);
	print_k("cudaOccupancyMaxActiveBlocksPerMultiprocessor");
	CHECK_RT(	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, p_foo, 1, dynamicSmemSize)		);
	print_k("cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
	CHECK_RT(	cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&numBlocks, p_foo, 256, dynamicSmemSize, cudaOccupancyDisableCachingOverride)		);
	//print_k("cudaOccupancyMaxActiveClusters");
	//CHECK_RT(	cudaOccupancyMaxActiveClusters(&numClusters, p_foo, &launchConfig)		);			not supported
	/* end of Occupancy */

	/* start of Memory Management */
	float *hArr, *uvmArr;
	unsigned int hArrFlags;

	print_k("cudaHostAlloc");
	CHECK_RT(	cudaHostAlloc((void**)&hArr, 1024*1024*4, cudaHostAllocDefault)		);
	print_k("cudaHostGetFlags");
	CHECK_RT(	cudaHostGetFlags(&hArrFlags, hArr)		);
	print_k("cudaFreeHost");
	CHECK_RT(	cudaFreeHost(hArr)		);
	hArr = (float*)malloc(1024*1024*4);
	print_k("cudaHostRegister");
	CHECK_RT(	cudaHostRegister(hArr, 1024*1024*4, cudaHostRegisterDefault)		);
	print_k("cudaHostUnregister");
	CHECK_RT(	cudaHostUnregister(hArr)		);
	cudaPitchedPtr dPitchedArr;
	cudaExtent	extend3D = make_cudaExtent(1024,1024,4); 
	print_k("cudaMalloc3D");
	CHECK_RT(	cudaMalloc3D(&dPitchedArr, extend3D)		);
	//cudaArray_t dArray;
	//cudaChannelFormatDesc arrDesc; arrDesc.x=1024; arrDesc.y=1; arrDesc.z=1; arrDesc.w=1; arrDesc.f=cudaChannelFormatKindFloat;
	//print_k("cudaMallocArray");
	//CHECK_RT(	cudaMallocArray(&dArray, &arrDesc, 1024, 0, 0)		);
	print_k("cudaMallocManaged");
	CHECK_RT(	cudaMallocManaged((void**)&uvmArr, 1024*1024*256)		);
	print_k("cudaMemAdvise SetPreferredLocation");
	CHECK_RT(	cudaMemAdvise(uvmArr, 1024*1024*256, cudaMemAdviseSetPreferredLocation, 0)		);
	print_k("cudaMemAdvise SetAccessedBy");
	CHECK_RT(	cudaMemAdvise(uvmArr, 1024*1024*256, cudaMemAdviseSetAccessedBy, 1)		);
	print_k("cudaMemGetInfo");
	size_t memFree, memTotal;
	CHECK_RT(	cudaMemGetInfo(&memFree, &memTotal)		);
	print_k("cudaMalloc");
	CHECK_RT(	cudaMalloc((void**)&arr, sizeof(float)*4096)		);
	print_k("cudaMemcpy");
	CHECK_RT(	cudaMemcpy(arr,hArr, 1024, cudaMemcpyDefault)		);
	print_k("cudaMemcpyAsync Pri");
	CHECK_RT(	cudaMemcpyAsync(arr,hArr, 1024, cudaMemcpyDefault, hStreamPri)		);
	print_k("cudaMemcpyAsync 0");
	CHECK_RT(	cudaMemcpyAsync(arr,hArr, 1024, cudaMemcpyDefault, 0)		);
	print_k("cudaMemset");
	CHECK_RT(	cudaMemset(arr, 0, 1024)	);
	print_k("cudaMemsetAsync Pri");
	CHECK_RT(	cudaMemsetAsync(arr, 1, 1024, hStreamPri)	);
	print_k("cudaMemsetAsync 0");
	CHECK_RT(	cudaMemsetAsync(arr, 1, 1024, 0)	);
	print_k("cudaFree");
	CHECK_RT(	cudaFree(arr)		);
	/* end of Memory Management */
	
	/* start of Stream Ordered Memory Allocator */
	print_k("cudaMemPoolCreate");
	cudaMemPool_t mem_pool; cudaMemPoolProps poolProps; memset(&poolProps, 0, sizeof(cudaMemPoolProps)); 
	poolProps.allocType=cudaMemAllocationTypePinned; poolProps.handleTypes=cudaMemHandleTypePosixFileDescriptor; poolProps.location.type=cudaMemLocationTypeDevice; poolProps.location.id=0;

	CHECK_RT(	cudaMemPoolCreate(&mem_pool, &poolProps)		);
	float *dArrPool;
	print_k("cudaMallocFromPoolAsync");
	CHECK_RT(	cudaMallocFromPoolAsync((void**)&dArrPool,1024*4, memPool, hStreamPri)		);
	print_k("cudaFreeAsync");
	CHECK_RT(	cudaFreeAsync(dArrPool,	hStreamPri)		);
	print_k("cudaMallocFromPoolAsync");
	CHECK_RT(	cudaMallocFromPoolAsync((void**)&dArrPool,1024*4, memPool, hStreamPri)		);	
	print_k("cudaMallocAsync");
	CHECK_RT(	cudaMallocAsync((void**)&arr, 1024*1024*4, hStreamPri)		);
	/* end of Stream Ordered Memory Allocator */

	/* start of Texture Object Management */
	/* end of Texture Object Management */

	/* start of Surface Object Management */
	/* end of Surface Object Management */



	
	print_k("cudaDeviceSynchronize");
	CHECK_RT(	cudaDeviceSynchronize()		);
	print_k("Analysis end");
	return 0;	
}
