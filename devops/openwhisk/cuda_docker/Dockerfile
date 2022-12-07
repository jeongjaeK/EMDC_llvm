#use docker image :latest
FROM sunginh/emdc-ow-cuda-11.8.0:v0.3

RUN apt update
RUN apt install -y build-essential python3 python3-pip jq
RUN pip3 install --upgrade pip setuptools six
RUN pip3 install --no-cache-dir gevent flask requests

ENV FLASK_PROXY_PORT 8080
#ENV gpus all
#ENV privileged true
#ENV device /dev/dri

#ADD https://raw.githubusercontent.com/apache/incubator-openwhisk-runtime-docker/master/core/actionProxy/actionproxy.py /actionProxy/actionproxy.py

RUN mkdir -p /action
RUN mkdir -p /init
RUN mkdir -p /actionProxy

ADD actionproxy.py /actionProxy/actionproxy.py
ADD stub.sh /action/exec
ADD init.sh /init/init.sh

RUN chmod +x /action/exec
RUN chmod +x /init/init.sh

CMD ["/bin/bash", "-c", "cd /actionProxy && python3 -u actionproxy.py"]
