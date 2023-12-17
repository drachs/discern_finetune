docker build -t vllm -f Dockerfile.vllm .
docker run -it --gpus all -p 443:443 vllm
