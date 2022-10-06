#include <iostream>
#include <cuda.h>

static inline void
checkDrvError(CUresult res, const char *tok, const char *file, unsigned line)
{
	if( res != CUDA_SUCCESS){
		const char *errStr = NULL;
		(void)cuGetErrorString(res, &errStr);
		std::cerr << file << ":" << line << ' ' << tok << "failed (" << (unsigned)res << "): " << errStr << "\n";
	}
}
