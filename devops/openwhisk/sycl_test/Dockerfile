FROM sunginh/emdc-oneapi-base-cuda

RUN pip install --upgrade pip setuptools six
RUN pip install --no-cache-dir gevent flask requests
RUN apt install jq -y

ENV FLASK_PROXY_PORT 8080

ADD ./actionproxy.py /actionProxy/actionproxy.py

RUN mkdir -p /action
ADD stub.sh /action/exec

RUN chmod +x /action/exec

CMD ["/bin/bash", "-c", "cd actionProxy && python -u actionproxy.py"]
