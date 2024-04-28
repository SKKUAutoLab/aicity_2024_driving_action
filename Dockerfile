FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Install linux packages
ENV DEBIAN_FRONTEND noninteractive
RUN apt update
RUN TZ=Etc/UTC apt install -y tzdata
RUN apt install --no-install-recommends -y gcc git zip curl htop libgl1-mesa-glx libglib2.0-0 libpython3-dev gnupg g++

# Security updates
# https://security.snyk.io/vuln/SNYK-UBUNTU1804-OPENSSL-3314796
RUN apt upgrade --no-install-recommends -y openssl tar

# Create working directory
RUN mkdir -p /usr/src/aic24-track_3

# Install pip packages
RUN alias python=python3
RUN python3 -m pip install --upgrade pip wheel
# RUN pip install --no-cache albumentations comet gsutil click
# RUN pip install --no-cache torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

# Install requirements
COPY . /usr/src/aic24-track_3
# ENV FORCE_CUDA="1"
# ARG TORCH_CUDA_ARCH_LIST="3.7;5.0;5.2;6.0;6.1+PTX;7.0;7.5+PTX;8.0;8.6+PTX"
# ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
RUN pip install --no-cache --root-user-action=ignore -r /usr/src/aic24-track_3/requirements.txt
RUN pip install /usr/src/aic24-track_3/detectron2-0.6-cp310-cp310-linux_x86_64.whl
# Make workspace
WORKDIR /usr/src/aic24-track_3

# Copy contents
# COPY . /usr/src/app  (issues as not a .git directory)

# Set environment variables
ENV OMP_NUM_THREADS=1

# Cleanup
ENV DEBIAN_FRONTEND teletype
