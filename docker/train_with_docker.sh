#!/bin/bash

# Define the Docker image name and tag
IMAGE_NAME="pix2pix"
IMAGE_TAG="latest"
DOCKER_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

# Check if the Docker image exists
if ! docker images "${IMAGE_NAME}:${IMAGE_TAG}" | grep -q "${IMAGE_NAME}.*${IMAGE_TAG}"; then
    echo "Docker image '${DOCKER_IMAGE}' not found. Creating the image..."
    docker build -t "${DOCKER_IMAGE}" docker/
fi

# Check if the path parameter is provided
if [ "$#" -eq 0 ]; then
    echo "Error: Path parameter not provided. Usage: $0 --config-file /path/to/their/test_config_file_name"
    exit 1
fi

pix2pix_DIR="$(pwd)"
docker run -it --gpus all -v "${pix2pix_DIR}:/pix2pix" "${DOCKER_IMAGE}" python3 train.py "$@" 
