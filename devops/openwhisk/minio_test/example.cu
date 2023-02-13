#include <client.h>
#include <cuda.h>
#include <string>
#include <iostream>
#include <fstream>
#include <bits/stdc++.h>
#include <sstream>

__global__ void times2(int* in, int* res){
	res[threadIdx.x] = in[threadIdx.x]*2;	
}

int main(int argc, char* argv[]) {

	// Check how to use
	if(argc != 5){
		std::cout<<"How to use...\n" \
					"./exec [in bucket name] [in object name] [res bucket name] [res object name]\n";
		return -2;
	}
	// Create S3 base URL.
	minio::s3::BaseUrl base_url("115.145.175.44:9000");

	// Create credential provider.
	minio::creds::StaticProvider provider("arcs", "12341234");

	// Create S3 client.
	minio::s3::Client client(base_url, &provider);

	minio::s3::GetObjectArgs args;
	args.bucket.append(argv[1]);
	args.object.append(argv[2]);

	args.datafunc = [](minio::http::DataFunctionArgs args) -> bool {
		return true;	
	};
	
	minio::s3::GetObjectResponse resp = client.GetObject(args);
	
	if (resp){
	}
	else {
		std::cout << "{\"Download Error\": \"" << resp.Error().String() << "\"}";
		return -1;
	}

	// parse object
	std::stringstream ss(resp.data);
	size_t size;
	ss >> size;
	size_t alloc_size = sizeof(float) * size;

	// allocate host and GPU memory
	int *h_arr, *h_out;
	h_arr = (int*)malloc(alloc_size);
	h_out = (int*)malloc(alloc_size);
	int *arr, *out;
	cudaMalloc((void**)&arr, alloc_size);
	cudaMalloc((void**)&out, alloc_size);

	for(int i=0; i<size; ++i){
		ss >> h_arr[i];	
	}

	cudaMemcpy(arr, h_arr, alloc_size, cudaMemcpyHostToDevice);
	times2<<<1,32>>>(arr, out);
	cudaDeviceSynchronize();
	cudaMemcpy(h_out, out, alloc_size, cudaMemcpyDeviceToHost);

	// put the result	
	FILE *file = fopen(argv[4], "wb");
	fwrite(h_out, sizeof(int), size, file);
	fclose(file);

	minio::s3::UploadObjectArgs up_args;
	up_args.bucket.append(argv[3]);
	up_args.object.append(argv[4]);
	up_args.filename.append(argv[4]);

	minio::s3::UploadObjectResponse up_resp = client.UploadObject(up_args);
	if(up_resp){}
	else{
		std::cout << "{\"Upload Error\": \"" << up_resp.Error().String() << "\"}";	
		return -1;
	}

	cudaFree(arr);
	cudaFree(out);
	free(h_arr);
	free(h_out);

	return 0;
}
