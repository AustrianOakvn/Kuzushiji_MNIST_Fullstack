FROM ubuntu:18.04

ENV PATH = "/root/miniconda3/bin:${PATH}"
ARG PATH = "/root/miniconda3/bin/${PATH}"

RUN apt update && apt install -y python3-dev wget

RUN wget  https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh \
    && mkdir root/.conda \
    && sh Miniconda3-py37_4.12.0-Linux-x86_64.sh -b \
    && rm -f Miniconda3-py37_4.12.0-Linux-x86_64.sh

# TODO: Put the environment name in here
RUN conda create -y -n env_name python=3.7

RUN /bin/bash -c "cd src && source activate env_name && pip install -r requirements.txt"