#!/bin/bash

CONTAINER_STORE="251264118427.dkr.ecr.us-west-1.amazonaws.com"

aws ecr get-login-password --region us-west-1 | sudo docker login --username AWS --password-stdin ${CONTAINER_STORE}
docker tag vllm:latest ${CONTAINER_STORE}/discern_vllm:latest
docker push ${CONTAINER_STORE}/discern_vllm:latest