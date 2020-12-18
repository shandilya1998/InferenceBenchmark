FROM 4/1AY0e-g59OebC4ujU_-G2nDfi8ZyglizONVnN5Q69S4RH_f-Ccr8tu0FVnGE

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

WORKDIR /root
ENV HOME /root

RUN apt-get update

RUN apt-get install -y --no-install-recommends \
    git \
    tar \
    build-essential \
    software-properties-common \
    ca-certificates \
    wget \
    curl \
    htop \
    zip \
    unzip \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libsqlite3-dev \
    libreadline-dev \
    libffi-dev \
    wget \
    libbz2-dev

RUN wget https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz && \
    tar -xf Python-3.7.5.tgz && \
    cd Python-3.7.5 && \
    ./configure && \
    make install && \
    python3 --version

COPY ./requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

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
