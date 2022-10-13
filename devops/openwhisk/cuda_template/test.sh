DOCKER_IMAGE=sunginh/emdc-ow-cuda-11.8.0:v0.2
#enter action name here
ACTION_NAME=cuda_hello		
ACTION=Action.zip

# exec,shell script for pre/post-processing, will be invoked by action
# exec.exe is executable

# zip a new action files
zip $ACTION exec exec.exe

# delete the action
wsk action delete $ACTION_NAME

# create a new action using custom docker image
wsk action create -i $ACTION_NAME --docker $DOCKER_IMAGE $ACTION

# invoke the action
wsk action invoke -i $ACTION_NAME --result 
