FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

ENV PYTHON_VERSION=3.7
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

WORKDIR /root
ENV HOME /root

RUN apt-get update

RUN apt-get install -y --no-install-recommends \
      git \
      build-essential \
      software-properties-common \
      ca-certificates \
      wget \
      curl \
      htop \
      zip \
      unzip

RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt install python3.7 python3-pip && \
    python3 -m pip --no-cache-dir install --upgrade pip setuptools && \
    ln -s $(which python3) /usr/local/bin/python

COPY .requirements.txt ./requirements.txt
COPY ./setup_env.sh ./setup_env.sh
RUN chmod +x setup_env.sh && \
    pip install --upgrade pip \
    sh setup_env.sh \
    ln -s /usr/local/cuda/lib64 /usr/local/cuda/lib && \
    cp /usr/lib/x86_64-linux-gnu/libnccl* /usr/local/cuda/lib && \
    ldconfig

RUN  echo "/usr/local/cuda/compat" >> /etc/ld.so.conf.d/cuda-10-0.conf && \
    ldconfig

ENTRYPOINT ["/bin/bash"]
