#use docker image :latest
FROM sunginh/emdc-oneapi-base-cuda

RUN pip install --upgrade pip setuptools six
RUN pip install --no-cache-dir gevent flask requests

ENV FLASK_PROXY_PORT 8080
#ENV gpus all
#ENV privileged true
#ENV device /dev/dri

#ADD https://raw.githubusercontent.com/apache/incubator-openwhisk-runtime-docker/master/core/actionProxy/actionproxy.py /actionProxy/actionproxy.py

#RUN mkdir -p /action
ADD stub.sh /action/exec

RUN chmod +x /action/exec

CMD ["/bin/bash", "-c", "cd actionProxy && python -u actionproxy.py"]
