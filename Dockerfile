FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel as base
FROM ubuntu:22.04

RUN apt-get update -y
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y install python3.10
RUN apt-get -y install pip git vim curl libosmesa6-dev libgl1-mesa-glx libglfw3

RUN mkdir -p /root/.mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C /root/.mujoco \
    && rm mujoco.tar.gz

ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}

ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt

