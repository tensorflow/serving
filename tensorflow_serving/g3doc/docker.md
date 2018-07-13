# Using TensorFlow Serving via Docker

One of the easiest ways to get started using TensorFlow Serving is via
[Docker](http://www.docker.com/).

## Installing Docker

General installation instructions are
[on the Docker site](https://docs.docker.com/install/), but we give some quick
links here:

*   [Docker for macOS](https://docs.docker.com/docker-for-mac/install/)
*   [Docker for Windows](https://docs.docker.com/docker-for-windows/install/)
    for Windows 10 Pro or later
*   [Docker Toolbox](https://docs.docker.com/toolbox/) for much older versions
    of macOS, or versions of Windows before Windows 10 Pro

## Serving with Docker

### Pulling a serving image

Once you have Docker installed, you can pull the latest TensorFlow Serving
docker image by running:

```shell
docker pull tensorflow/serving
```

This will pull down an minimal Docker image with TensorFlow Serving installed.

See the Docker Hub [tensorflow/serving
repo](http://hub.docker.com/r/tensorflow/serving/tags/) for other versions of
images you can pull.

### Serving example

Once you have pulled the serving image, you can try serving an example model.

We will use a toy model called `Half Plus Three`, which will predict values `0.5 * x + 3`
for the values we provide for prediction.

To get this model, first clone the TensorFlow Serving repo.

```shell
mkdir -p /tmp/tfserving
cd /tmp/tfserving
git clone --depth=1 https://github.com/tensorflow/serving
```

Next, run the TensorFlow Serving container pointing it to this model and opening
the REST API port (8501):

```shell
docker run -p 8501:8501 \
-v /tmp/tfserving/serving/tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_three:/models/half_plus_three \
-e MODEL_NAME=half_plus_three -t tensorflow/serving &
```

This will run the docker container and launch the TensorFlow Serving Model
Server, bind the REST API port 8501, and map our desired model from our host to
where models are expected in the container. We also pass the name of the model
as an environment variable, which will be important when we query the model.

To query the model using the predict API, you can run

```shell
curl -d '{"instances": [1.0, 2.0, 5.0]}' -X POST http://localhost:8501/v1/models/half_plus_three:predict
```

NOTE: Older versions of Windows and other systems without curl can download it
[here](https://curl.haxx.se/download.html).

This should return a set of values:

```json
{ "predictions": [3.5, 4.0, 5.5] }
```

More information on using the RESTful API can be found [here](api_rest.md).

## Developing with Docker

### Pulling a development image

For a development environment where you can build TensorFlow Serving, you can
try:

```shell
docker pull tensorflow/serving:latest-devel
```

For a development environment where you can build TensorFlow Serving with GPU
support, use:

```shell
docker pull tensorflow/serving:latest-devel-gpu
```

See the Docker Hub [tensorflow/serving
repo](http://hub.docker.com/r/tensorflow/serving/tags/) for other versions of
images you can pull.

### Development example

After pulling one of the development Docker images, you can run it while opening
the gRPC port (8500):

```shell
docker run -it -p 8500:8500 tensorflow/serving:latest-devel
```

#### Testing the development environment

To test a model, from inside the container try:

```shell
bazel build -c opt //tensorflow_serving/example:mnist_saved_model
# train the mnist model
bazel-bin/tensorflow_serving/example/mnist_saved_model /tmp/mnist_model
# serve the model
tensorflow_model_server --port=8500 --model_name=mnist --model_base_path=/tmp/mnist_model/ &
# build the client
bazel build -c opt //tensorflow_serving/example:mnist_client
# test the client
bazel-bin/tensorflow_serving/example/mnist_client --num_tests=1000 --server=localhost:8500
```

## Dockerfiles

We currently maintain the following Dockerfiles:

*   [`Dockerfile`](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/tools/docker/Dockerfile),
    which is a minimal VM with TensorFlow Serving installed.

*   [`Dockerfile.devel`](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/tools/docker/Dockerfile.devel),
    which is a minimal VM with all of the dependencies needed to build
    TensorFlow Serving.

*   [`Dockerfile.devel-gpu`](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/tools/docker/Dockerfile.devel),
    which is a minimal VM with all of the dependencies needed to build
    TensorFlow Serving with GPU support.

### Building a container from a Dockerfile

If you'd like to build your own Docker image from a Dockerfile, you can do so by
running the Docker build command:

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

TIP: Before attempting to build an image, check the Docker Hub
[tensorflow/serving repo](http://hub.docker.com/r/tensorflow/serving/tags/) to
make sure an image that meets your needs doesn't already exist.

By default, building TensorFlow Serving from sources consumes a lot of RAM. If
RAM is an issue on your system, you may limit RAM usage by specifying
`--local_resources 2048,.5,1.0` while invoking bazel. You can use this same
mechanism to tweak the optmizations you're building TensorFlow Serving with. For
example:

```shell
docker build --pull --build-arg TF_SERVING_BUILD_OPTIONS="--copt=-mavx \
--cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0 --local_resources 2048,.5,1.0" -t \
$USER/tensorflow-serving-devel -f Dockerfile.devel .
```

### Running a container

This assumes you have built the `Dockerfile.devel` container.

To run the container opening the gRPC port (8500):

```shell
docker run -it -p 8500:8500 $USER/tensorflow-serving-devel
```

From here, you can follow the instructions for
[testing a development environment](#testing-the-development-environment).
