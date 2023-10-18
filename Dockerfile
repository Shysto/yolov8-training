FROM busybox as staging

RUN mkdir /tmp/files
COPY ./requirements.txt /tmp/files/requirements.txt

# ---------------------------------------------------

FROM nvidia/cuda:11.3.1-base-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHON_VERSION=3.10.6

WORKDIR /root

RUN apt-get update && apt-get upgrade -y && \
    apt install -y \
    # liblzma-dev for python pandas
    sudo tzdata vim git cmake wget curl unzip tar build-essential libbz2-dev tk-dev libssh-dev dumb-init liblzma-dev && \
    # set timezone to JST
    ln -fs /usr/share/zoneinfo/Asia/Tokyo /etc/localtime && dpkg-reconfigure -f noninteractive tzdata && \
    # create user
    groupadd -r trainer --gid=999 && \
	useradd -g trainer --uid=999 --shell=/bin/bash --create-home trainer && \
    # add user to the sudoers without password for nvidia-smi
    echo "trainer ALL=(ALL) NOPASSWD: /usr/bin/nvidia-smi" > /etc/sudoers.d/trainer && \
    chmod 0440 /etc/sudoers.d/trainer && \
    # apt clean
    apt clean && rm -fr /var/lib/apt/lists/

# compile python 3.10.6
RUN mkdir -p /tmp/build && cd /tmp/build && \
    apt-get update && apt install -y --no-install-recommends make cmake gcc git g++ unzip wget build-essential zlib1g-dev libffi-dev libssl-dev && \
    apt clean && rm -fr /var/lib/apt/lists/ && \
    wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar zxf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --enable-optimizations && \
    make altinstall && \
    ln -s /usr/local/bin/python3.10 /bin/python3 && \
    ln -s /usr/local/bin/pip3.10 /bin/pip3 && \
    cd / && rm -fr /tmp/build

COPY --from=staging --chown=trainer:trainer /tmp/files /tmp/files

USER trainer

RUN cd ${HOME} && \
    python3 -m venv venv && \
    venv/bin/python3 -m pip install --upgrade pip && \
    venv/bin/python3 -m pip install --no-cache-dir torch>=1.7 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    venv/bin/python3 -m pip install --no-cache-dir -r /tmp/files/requirements.txt && \
    rm -fr ${HOME}/.cache/ /tmp/files

COPY --chown=trainer:trainer ./src /app/src

WORKDIR /app

ENTRYPOINT ["/usr/bin/dumb-init", "-c", "--"]

CMD bash -c "source /home/trainer/venv/bin/activate"
