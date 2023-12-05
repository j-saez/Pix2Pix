FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

WORKDIR /pix2pix
COPY requirements.txt /pix2pix


RUN apt update && apt install python3-pip -y
RUN cat /pix2pix/requirements.txt
RUN pip install -r requirements.txt
