# syntax=docker/dockerfile:1

FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates python3.12 python3.12-venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN python3.12 -m venv /opt/usearch/venv
ENV PATH="/opt/usearch/venv/bin:$PATH"
RUN python -m pip install --no-cache-dir --upgrade "pip>=26.2.2" && \
    python -m pip install --no-cache-dir ucall usearch

WORKDIR /usearch
COPY python/usearch/server.py server.py

ENTRYPOINT ["python3", "./server.py"]
EXPOSE 8545
