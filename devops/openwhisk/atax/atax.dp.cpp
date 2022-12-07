/**
 * atax.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>

#include "polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.5

#define GPU_DEVICE 0

/* Problem size. */
#define NX 16384
//#define NX 4096
#define NY 16384
//#define NY 4096

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 256
#define DIM_THREAD_BLOCK_Y 1

#ifndef M_PI
#define M_PI 3.14159
#endif

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



void init_array(DATA_TYPE *x, DATA_TYPE *A)
{
	int i, j;

	for (i = 0; i < NX; i++)
	{
		x[i] = i * M_PI;
		for (j = 0; j < NY; j++)
		{
			A[i*NY + j] = ((DATA_TYPE) i*(j)) / NX;
		}
	}
}


void compareResults(DATA_TYPE *z, DATA_TYPE *z_outputFromGpu)
{
	int i, fail;
	fail = 0;

	for (i=0; i<NY; i++)
	{
		if (percentDiff(z[i], z_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
		}		
	}
	
	// print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void GPU_argv_init()
{
//	cudaDeviceProp deviceProp;
//	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
//	printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
//	cudaSetDevice( GPU_DEVICE );
}


void atax_kernel1(DATA_TYPE *A, DATA_TYPE *x, DATA_TYPE *tmp,
                  sycl::nd_item<3> item_ct1)
{
        int i = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                item_ct1.get_local_id(2);

        if (i < NX)
	{
		int j;
		DATA_TYPE acc = tmp[i];

		for(j=0; j < NY; j++)
		{
			acc += A[i * NY + j] * x[j];
		}

		tmp[i] = acc;
	}
}

void atax_kernel2(DATA_TYPE *A, DATA_TYPE *y, DATA_TYPE *tmp,
                  sycl::nd_item<3> item_ct1)
{
        int j = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                item_ct1.get_local_id(2);

        if (j < NY)
	{
		int i;
		DATA_TYPE acc = y[j];
		for(i=0; i < NX; i++)
		{
			acc += A[i * NY + j] * tmp[i];
		}
		y[j] = acc;
	}
}


void atax_cpu(DATA_TYPE* A, DATA_TYPE* x, DATA_TYPE* y, DATA_TYPE* tmp)
{
	int i,j;
	
	for (i= 0; i < NY; i++)
	{
    	y[i] = 0;
	}
  
	for (i = 0; i < NX; i++)
 	{
      	tmp[i] = 0;

      	for (j = 0; j < NY; j++)
		{
			tmp[i] = tmp[i] + A[i*NY + j] * x[j];
		}
		
      	for (j = 0; j < NY; j++)
		{
			y[j] = y[j] + A[i*NY + j] * tmp[i];
		}
    }
}


void ataxGpu(DATA_TYPE* A, DATA_TYPE* x, DATA_TYPE* y, DATA_TYPE* tmp, DATA_TYPE* y_outputFromGpu)
{
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
        double t_start, t_end;

	DATA_TYPE *A_gpu;
	DATA_TYPE *x_gpu;
	DATA_TYPE *y_gpu;
	DATA_TYPE *tmp_gpu;

        A_gpu = (DATA_TYPE *)sycl::malloc_device(sizeof(DATA_TYPE) * NX * NY,
                                                 q_ct1);
        x_gpu = sycl::malloc_device<DATA_TYPE>(NY, q_ct1);
        y_gpu = sycl::malloc_device<DATA_TYPE>(NY, q_ct1);
        tmp_gpu = sycl::malloc_device<DATA_TYPE>(NX, q_ct1);

        q_ct1.memcpy(A_gpu, A, sizeof(DATA_TYPE) * NX * NY);
        q_ct1.memcpy(x_gpu, x, sizeof(DATA_TYPE) * NY);
        q_ct1.memcpy(y_gpu, y, sizeof(DATA_TYPE) * NY);
        q_ct1.memcpy(tmp_gpu, tmp, sizeof(DATA_TYPE) * NX).wait();

        sycl::range<3> block(1, DIM_THREAD_BLOCK_Y, DIM_THREAD_BLOCK_X);
        sycl::range<3> grid1(1, 1, (size_t)(ceil(((float)NX) / ((float)block[2]))));
        sycl::range<3> grid2(1, 1, (size_t)(ceil(((float)NY) / ((float)block[2]))));

        t_start = rtclock();
        /*
        DPCT1049:0: The workgroup size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the workgroup size if needed.
        */
        q_ct1.parallel_for(sycl::nd_range<3>(grid1 * block, block),
                           [=](sycl::nd_item<3> item_ct1) {
                                   atax_kernel1(A_gpu, x_gpu, tmp_gpu,
                                                item_ct1);
                           });
        dev_ct1.queues_wait_and_throw();
        /*
        DPCT1049:1: The workgroup size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the workgroup size if needed.
        */
        q_ct1.parallel_for(sycl::nd_range<3>(grid2 * block, block),
                           [=](sycl::nd_item<3> item_ct1) {
                                   atax_kernel2(A_gpu, y_gpu, tmp_gpu,
                                                item_ct1);
                           });
        dev_ct1.queues_wait_and_throw();
        t_end = rtclock();
	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

        q_ct1.memcpy(y_outputFromGpu, y_gpu, sizeof(DATA_TYPE) * NX).wait();

        sycl::free(A_gpu, q_ct1);
        sycl::free(x_gpu, q_ct1);
        sycl::free(y_gpu, q_ct1);
        sycl::free(tmp_gpu, q_ct1);
}


int main(int argc, char** argv)
{
	double t_start, t_end;

	DATA_TYPE* A;
	DATA_TYPE* x;
	DATA_TYPE* y;
	DATA_TYPE* y_outputFromGpu;
	DATA_TYPE* tmp;

	A = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));
	x = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
	y = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
	y_outputFromGpu = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
	tmp = (DATA_TYPE*)malloc(NX*sizeof(DATA_TYPE));

	init_array(x, A);

	GPU_argv_init();
	ataxGpu(A, x, y, tmp, y_outputFromGpu);
	
//	t_start = rtclock();
//	atax_cpu(A, x, y, tmp);
//	t_end = rtclock();
//	fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

//	compareResults(y, y_outputFromGpu);

	free(A);
	free(x);
	free(y);
	free(y_outputFromGpu);
	free(tmp);

  	return 0;
}

