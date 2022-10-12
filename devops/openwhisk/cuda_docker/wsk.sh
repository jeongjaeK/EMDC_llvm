DOCKER_IMAGE=sunginh/emdc-oneapi-base-cuda
#DOCKER_IMAGE=openwhisk/dockerskeleton
ACTION_NAME=sycl_test
#ACTION=Action.zip
ACTION=test.zip

wsk action delete $ACTION_NAME
wsk action create -i $ACTION_NAME --docker $DOCKER_IMAGE $ACTION
#wsk action create -i $ACTION_NAME $ACTION --native
wsk action invoke -i $ACTION_NAME --result
