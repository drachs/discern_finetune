docker build -t vllm -f Dockerfile.vllm .
docker run -it --gpus all -p 8000:8000 vllm
