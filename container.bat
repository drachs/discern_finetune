# Pytorch 1.x
#docker run --gpus all -d -it -p 8848:8888 -v .:/home/jovyan/work -e GRANT_SUDO=yes -e JUPYTER_ENABLE_LAB=yes --user root cschranz/gpu-jupyter:v1.5_cuda-11.6_ubuntu-20.04_python-only

#docker run --gpus all -d -it -p 8848:8888 -v .:/home/jovyan/work -e GRANT_SUDO=yes -e JUPYTER_ENABLE_LAB=yes --user root cschranz/gpu-jupyter:v1.5_cuda-11.6_ubuntu-20.04_python-only

#echo run `jupyter server list` from inside the container to see the token

docker build -t pytorch2 -f Dockerfile.Pytorch2 .
docker run -it --gpus all -p 8888:8888 -v .:/opt/notebooks pytorch2
