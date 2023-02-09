#include <client.h>

int main(int argc, char* argv[]) {
  // Create S3 base URL.
  minio::s3::BaseUrl base_url("115.145.175.44:9000");

  // Create credential provider.
  minio::creds::StaticProvider provider("arcs", "12341234");

  // Create S3 client.
  minio::s3::Client client(base_url, &provider);

  std::string bucket_name = "ow-test00";

  // Check 'asiatrip' bucket exist or not.
  bool exist;
  {
    minio::s3::BucketExistsArgs args;
    args.bucket = bucket_name;

    minio::s3::BucketExistsResponse resp = client.BucketExists(args);
    if (!resp) {
      std::cout << "unable to do bucket existence check; " << resp.Error()
                << std::endl;
      return EXIT_FAILURE;
    }

    exist = resp.exist;
  }

  // Make 'asiatrip' bucket if not exist.
  if (!exist) {
    minio::s3::MakeBucketArgs args;
    args.bucket = bucket_name;

    minio::s3::MakeBucketResponse resp = client.MakeBucket(args);
    if (!resp) {
      std::cout << "unable to create bucket; " << resp.Error() << std::endl;
      return EXIT_FAILURE;
    }
  }



  // Upload '/home/user/Photos/asiaphotos.zip' as object name
  // 'asiaphotos-2015.zip' to bucket 'asiatrip'.
  minio::s3::UploadObjectArgs args;
  args.bucket = bucket_name;
  args.object = "test0.txt";
  args.filename = "/home/sihong/test.txt";

  minio::s3::UploadObjectResponse resp = client.UploadObject(args);
  if (!resp) {
    std::cout << "unable to upload object; " << resp.Error() << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "'/home/sihong/test_data/test.txt' is successfully uploaded as "
            << "object 'test.txt' to bucket 'arcs'."
            << std::endl;

  return EXIT_SUCCESS;
}
