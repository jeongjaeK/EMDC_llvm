#use docker image :latest
FROM sunginh/emdc-ow-cuda-11.8.0:v0.1

RUN apt update
RUN apt install -y build-essential python3 python3-pip
RUN pip3 install --upgrade pip setuptools six
RUN pip3 install --no-cache-dir gevent flask requests

ENV FLASK_PROXY_PORT 8080
#ENV gpus all
#ENV privileged true
#ENV device /dev/dri

#ADD https://raw.githubusercontent.com/apache/incubator-openwhisk-runtime-docker/master/core/actionProxy/actionproxy.py /actionProxy/actionproxy.py

#RUN mkdir -p /action
ADD stub.sh /action/exec

RUN chmod +x /action/exec

CMD ["/bin/bash", "-c", "cd actionProxy && python3 -u actionproxy.py"]
