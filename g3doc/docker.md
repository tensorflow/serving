# Using TensorFlow Serving via Docker

This directory contains Dockerfiles to make it easy to get up and running with
TensorFlow Serving via [Docker](http://www.docker.com/).

## Installing Docker

General installation instructions are
[on the Docker site](https://docs.docker.com/installation/), but we give some
quick links here:

* [OSX](https://docs.docker.com/installation/mac/): [docker toolbox](https://www.docker.com/toolbox)
* [ubuntu](https://docs.docker.com/installation/ubuntulinux/)

## Which containers exist?

We currently maintain the following Dockerfiles:

* [`Dockerfile.devel`](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/tools/docker/Dockerfile.devel),
which is a minimal VM with all of the dependencies needed to build TensorFlow
Serving.

## Building a container
run;

```shell
docker build --pull -t $USER/tensorflow-serving-devel -f Dockerfile.devel .
```

## Running a container
This assumes you have built the container.

`Dockerfile.devel`: Use the development container to clone and test the
TensorFlow Serving repository.

Run the container;

```shell
docker run -it $USER/tensorflow-serving-devel
```

Clone, configure and test Tensorflow Serving in the running container;

```shell
git clone --recurse-submodules https://github.com/tensorflow/serving
cd serving/tensorflow
./configure
cd ..
bazel test tensorflow_serving/...
```

