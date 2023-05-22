#include <client.h>
#include <string>
#include <iostream>
#include <fstream>
#include <bits/stdc++.h>

int main(int argc, char* argv[]) {
	// Create S3 base URL.
	minio::s3::BaseUrl base_url("115.145.175.44:9000");

	// Create credential provider.
	minio::creds::StaticProvider provider("arcs", "12341234");

	// Create S3 client.
	minio::s3::Client client(base_url, &provider);

	minio::s3::DownloadObjectArgs args;
	args.bucket.append(argv[1]);
	std::cout<<args.bucket<<"\n";
	args.object.append(argv[2]);
	std::cout<<args.object<<"\n";
	args.filename.append(argv[3]);
	
	minio::s3::DownloadObjectResponse resp = client.DownloadObject(args);
	
	if (resp){
		std::cout << "data : " << resp.data << "\n";
	}
	else {
		std::cout << "unabled to get object; " << resp.Error().String() << "\n";	
	}


	return 0;
}
