# How-To: running benchmarks in a container

```sh
docker rm zk-benchmarks 
docker rmi zk-benchmarks-image 
docker system prune -a --force --volumes

docker build -t zk-benchmarks-image .

docker run -it --rm --gpus all --name zk-benchmarks -v "$(pwd)":/home/runner/zk-benchmarks -w /home/runner/zk-benchmarks zk-benchmarks-image

```
