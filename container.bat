docker build -t pytorch2 -f Dockerfile.Pytorch2 .
docker run -it --gpus all -p 8888:8888 -v .:/opt/notebooks pytorch2
