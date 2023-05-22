/**
 * 2DConvolution.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <iostream>
#include <string>
#include <fstream>

#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <cuda.h>

#include "polybenchUtilFuncts.h"

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

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

/* Problem size */
#define NI 16384
#define NJ 16384

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

#ifndef MALLOC
	#define HOST // default is cudaMallocHost
#endif

string output;

void conv2D(DATA_TYPE* A, DATA_TYPE* B)
{
	int i, j;
	DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +0.2;  c21 = +0.5;  c31 = -0.8;
	c12 = -0.3;  c22 = +0.6;  c32 = -0.9;
	c13 = +0.4;  c23 = +0.7;  c33 = +0.10;


	for (i = 1; i < NI - 1; ++i) // 0
	{
		for (j = 1; j < NJ - 1; ++j) // 1
		{
			B[i*NJ + j] = c11 * A[(i - 1)*NJ + (j - 1)]  +  c12 * A[(i + 0)*NJ + (j - 1)]  +  c13 * A[(i + 1)*NJ + (j - 1)]
				+ c21 * A[(i - 1)*NJ + (j + 0)]  +  c22 * A[(i + 0)*NJ + (j + 0)]  +  c23 * A[(i + 1)*NJ + (j + 0)] 
				+ c31 * A[(i - 1)*NJ + (j + 1)]  +  c32 * A[(i + 0)*NJ + (j + 1)]  +  c33 * A[(i + 1)*NJ + (j + 1)];
		}
	}
}



void init(DATA_TYPE* A)
{
	int i, j;

	for (i = 0; i < NI; ++i)
    	{
		for (j = 0; j < NJ; ++j)
		{
			A[i*NJ + j] = (float)rand()/RAND_MAX;
        	}
    	}
}


void compareResults(DATA_TYPE* B, DATA_TYPE* B_outputFromGpu)
{
	int i, j, fail;
	fail = 0;
	
	// Compare a and b
	for (i=1; i < (NI-1); i++) 
	{
		for (j=1; j < (NJ-1); j++) 
		{
			if (percentDiff(B[i*NJ + j], B_outputFromGpu[i*NJ + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				fail++;
			}
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
	
}


void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
	cudaSetDevice( GPU_DEVICE );
}


__global__ void Convolution2D_kernel(DATA_TYPE *A, DATA_TYPE *B)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +0.2;  c21 = +0.5;  c31 = -0.8;
	c12 = -0.3;  c22 = +0.6;  c32 = -0.9;
	c13 = +0.4;  c23 = +0.7;  c33 = +0.10;

	if ((i < NI-1) && (j < NJ-1) && (i > 0) && (j > 0))
	{
		B[i * NJ + j] =  c11 * A[(i - 1) * NJ + (j - 1)]  + c21 * A[(i - 1) * NJ + (j + 0)] + c31 * A[(i - 1) * NJ + (j + 1)] 
			+ c12 * A[(i + 0) * NJ + (j - 1)]  + c22 * A[(i + 0) * NJ + (j + 0)] +  c32 * A[(i + 0) * NJ + (j + 1)]
			+ c13 * A[(i + 1) * NJ + (j - 1)]  + c23 * A[(i + 1) * NJ + (j + 0)] +  c33 * A[(i + 1) * NJ + (j + 1)];
	}
}


void convolution2DCuda(DATA_TYPE* A, DATA_TYPE* B_outputFromGpu, int id)
{
	double t_start, t_end;

	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;

	t_start = rtclock();
	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NI * NJ);
	cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * NI * NJ);
	t_end = rtclock();

	output += to_string(t_end - t_start) + " ";		
//	fprintf(stdout, "%d: cudaMalloc: %0.6lfs\n", id, t_end - t_start);

	t_start = rtclock();
	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyHostToDevice);
	t_end = rtclock();
	
	output += to_string(t_end - t_start) + " ";		
	//fprintf(stdout, "%d: cudaMemcpy: %0.6lfs\n", id, t_end - t_start);

	t_start = rtclock();
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((size_t)ceil( ((float)NI) / ((float)block.x) ), (size_t)ceil( ((float)NJ) / ((float)block.y)) );
	Convolution2D_kernel<<<grid,block>>>(A_gpu,B_gpu);
	cudaDeviceSynchronize();
	t_end = rtclock();
	output += to_string(t_end - t_start) + " ";		
	//fprintf(stdout, "%d: kernel : %0.6lfs\n", id, t_end - t_start);

	t_start = rtclock();
	cudaMemcpy(B_outputFromGpu, B_gpu, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyDeviceToHost);	
	t_end = rtclock();
	output += to_string(t_end - t_start) + " ";		
	//fprintf(stdout, "%d: cudaMemcpy out: %0.6lfs\n", id, t_end - t_start);

	t_start = rtclock();
	CHECK_RT(	cudaFree(A_gpu) 	)
	CHECK_RT(	cudaFree(B_gpu) 	)
	t_end = rtclock();
	output += to_string(t_end - t_start) + " ";		
	//fprintf(stdout, "%d:, GPU Runtime: %0.6lfs\n", id,t_end - t_start);
}


int main(int argc, char *argv[])
{
	int new_job_arrived;
	cudaFree(0);
	double t_start, t_end, t_m_s, t_m_e;
	cudaDeviceSetLimit(cudaLimitStackSize, 0);
	int fd, bytesread;
	char buf[1024];

	while(1){
		//	scanf("%d",&new_job_arrived);
		//	new_job_arrived=0;
		fd = open("fifo.1", O_RDONLY);
		ofstream ofs("output");

		while(1){
			if( bytesread = read(fd, buf, 1024 ) > 0)
				break;
		}
		t_start = rtclock();
		output = "{\"msg\" : \"";
		//	std::cout<<"task: "<<new_job_arrived<<"\n";

		DATA_TYPE* A;
		//DATA_TYPE* B;  
		DATA_TYPE* B_outputFromGpu;

		t_m_s = rtclock();
#ifdef MALLOC
		A = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE));
		B_outputFromGpu = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE));
#endif
#ifdef HOST
		cudaMallocHost((void**)&A, NI*NJ*sizeof(DATA_TYPE));
		//B = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE));
		cudaMallocHost((void**)&B_outputFromGpu, NI*NJ*sizeof(DATA_TYPE));
#endif
		t_m_e = rtclock();
#ifdef MALLOC
		output += to_string(t_m_e - t_m_s) + " ";		
		//		fprintf(stdout, "%d: malloc : %0.6lfs\n", new_job_arrived, t_m_e - t_m_s);
#endif
#ifdef HOST
		output += to_string(t_m_e - t_m_s) + " ";		
		//		fprintf(stdout, "%d: cudaMallocHost : %0.6lfs\n", new_job_arrived, t_m_e - t_m_s);
#endif
		//initialize the arrays
		//	init(A);

		//	GPU_argv_init();

		convolution2DCuda(A, B_outputFromGpu,new_job_arrived);

		//	conv2D(A, B);
		//	compareResults(B, B_outputFromGpu);
#ifdef MALLOC
		free(A);
		//free(B);
		free(B_outputFromGpu);
#endif
#ifdef HOST
		CHECK_RT(	cudaFreeHost(A)		)	
		CHECK_RT(	cudaFreeHost(B_outputFromGpu)	)
#endif
		t_end = rtclock();

		output += to_string(t_end - t_start) + " ";		
		//		fprintf(stdout, "%d: E2E latency : %0.6lfs\n", new_job_arrived, t_end - t_start);
		//		fflush(stdout);
		output += "\"}";

		ofs << output <<std::endl;
		ofs.flush();
		ofs.close();
		close(fd);

	}

	return 0;
}

