# Guide to flask_client

cd /Users/$USER/Documents/tf_learning/new_docker
DOCKERFILE=/Users/$USER/Documents/repos/tf-serving/tensorflow_serving/tools/docker/Dockerfile.devel
cp $DOCKERFILE .
docker build --pull -t tfs-flask:0.1 -f Dockerfile.devel .


docker run --name=tfs_flask -p 18888:8888 -p 19000:9000 -p 15000:5000 -v /tmp/models:/tmp/models -it tfs-flask:0.1

