# syntax=docker/dockerfile:1

FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3.12 python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir --break-system-packages ucall usearch

WORKDIR /usearch
COPY python/usearch/server.py server.py

ENTRYPOINT ["python3", "./server.py"]
EXPOSE 8545
