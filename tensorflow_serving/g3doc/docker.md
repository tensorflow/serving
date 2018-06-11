# Using TensorFlow Serving via Docker

This directory contains Dockerfiles to make it easy to get up and running with
TensorFlow Serving via [Docker](http://www.docker.com/).

## Installing Docker

General installation instructions are
[on the Docker site](https://docs.docker.com/installation/), but we give some
quick links here:

*   [OSX](https://docs.docker.com/installation/mac/): [docker
    toolbox](https://www.docker.com/toolbox)
*   [Ubuntu](https://docs.docker.com/installation/ubuntulinux/)

## Which containers exist?

We currently maintain the following Dockerfiles:

*   [`Dockerfile`](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/tools/docker/Dockerfile),
    which is a minimal VM with TensorFlow Serving installed.

*   [`Dockerfile.devel`](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/tools/docker/Dockerfile.devel),
    which is a minimal VM with all of the dependencies needed to build
    TensorFlow Serving.

*   [`Dockerfile.devel-gpu`](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/tools/docker/Dockerfile.devel),
    which is a minimal VM with all of the dependencies needed to build
    TensorFlow Serving with GPU support.

## Building a container

`Dockerfile`:

```shell
docker build --pull -t $USER/tensorflow-serving .
```

`Dockerfile.devel`:

```shell
docker build --pull -t $USER/tensorflow-serving-devel -f Dockerfile.devel .
```

`Dockerfile.devel-gpu`:

```shell
docker build --pull -t $USER/tensorflow-serving-devel-gpu -f Dockerfile.devel-gpu .
```

## Running a container

### Serving Example

This assumes you have built the `Dockerfile` container.

First you will need a model. Clone the TensorFlow Serving repo.

```shell
mkdir -p /tmp/tfserving
cd /tmp/tfserving
git clone --recursive https://github.com/tensorflow/serving
```

We will use a toy model called Half Plus Three, which will predict values 0.5\*x
+ 3 for the values we provide for prediction.

```shell
docker run -p 8501:8501 \
-v /tmp/tfserving/serving/tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_three:/models/half_plus_three \
-e MODEL_NAME=half_plus_three -t $USER/tensorflow-serving &
```

This will run the docker container and launch the TensorFlow Serving Model
Server, bind the REST API port 8501, and map our desired model from our host to
where models are expected in the container. We also pass the name of the model
as an environment variable, which will be important when we query the model.

To query the model using the predict API, you can run

```shell
curl -d '{"instances": [1.0, 2.0, 5.0]}' -X POST http://localhost:8501/v1/models/half_plus_three:predict
```

This should return a set of values:

```json
{ "predictions": [3.5, 4.0, 5.5] }
```

More information on using the RESTful API can be found [here](api_rest.md).

### Development Example

This assumes you have built the `Dockerfile.devel` container.

To run the container:

```shell
docker run -it -p 8500:8500 $USER/tensorflow-serving-devel
```

This will pull the latest TensorFlow Serving release with a prebuilt binary and
working development environment. To test a model, from inside the container try:

``````shell
bazel build -c opt //tensorflow_serving/example:mnist_saved_model
# train the mnist model
bazel-bin/tensorflow_serving/example/mnist_saved_model /tmp/mnist_model
# serve the model
tensorflow_model_server --port=8500 --model_name=mnist --model_base_path=/tmp/mnist_model/ &
# build the client
bazel build -c opt //tensorflow_serving/example:mnist_client
# test the client
bazel-bin/tensorflow_serving/example/mnist_client --num_tests=1000 --server=localhost:8500
`````
``````
