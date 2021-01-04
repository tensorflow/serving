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

See the Docker Hub
[tensorflow/serving repo](http://hub.docker.com/r/tensorflow/serving/tags/) for
other versions of images you can pull.

### Development example

After pulling one of the development Docker images, you can run it while opening
the gRPC port (8500):

```shell
docker run -it -p 8500:8500 --gpus all tensorflow/serving:latest-devel
```

#### Testing the development environment

To test a model, from inside the container try:

```shell
# train the mnist model
python tensorflow_serving/example/mnist_saved_model.py /tmp/mnist_model
# serve the model
tensorflow_model_server --port=8500 --model_name=mnist --model_base_path=/tmp/mnist_model/ &
# test the client
python tensorflow_serving/example/mnist_client.py --num_tests=1000 --server=localhost:8500
```

## Dockerfiles

We currently maintain the following Dockerfiles:

*   [`Dockerfile`](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/tools/docker/Dockerfile),
    which is a minimal VM with TensorFlow Serving installed.

*   [`Dockerfile.gpu`](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/tools/docker/Dockerfile.gpu),
    which is a minimal VM with TensorFlow Serving with GPU support to be used
    with `nvidia-docker`.

*   [`Dockerfile.devel`](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/tools/docker/Dockerfile.devel),
    which is a minimal VM with all of the dependencies needed to build
    TensorFlow Serving.

*   [`Dockerfile.devel-gpu`](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/tools/docker/Dockerfile.devel-gpu),
    which is a minimal VM with all of the dependencies needed to build
    TensorFlow Serving with GPU support.

### Building a container from a Dockerfile

If you'd like to build your own Docker image from a Dockerfile, you can do so by
running the Docker build command:

`Dockerfile`:

```shell
docker build --pull -t $USER/tensorflow-serving .
```

`Dockerfile.gpu`:

```shell
docker build --pull -t $USER/tensorflow-serving-gpu -f Dockerfile.gpu .
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

Building from sources consumes a lot of RAM. If RAM is an issue on your system,
you may limit RAM usage by specifying `--local_ram_resources=2048` while
invoking Bazel. See the
[Bazel docs](https://docs.bazel.build/versions/master/user-manual.html#flag--local_{ram,cpu}_resources)
for more information. You can use this same mechanism to tweak the optmizations
you're building TensorFlow Serving with. For example:

```shell
docker build --pull --build-arg TF_SERVING_BUILD_OPTIONS="--copt=-mavx \
  --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0 --local_ram_resources=2048" -t \
  $USER/tensorflow-serving-devel -f Dockerfile.devel .
```

### Running a container

This assumes you have built the `Dockerfile.devel` container.

To run the container opening the gRPC port (8500):

```shell
docker run -it -p 8500:8500 $USER/tensorflow-serving-devel
```

TIP: If you're running a GPU image, be sure to run using the NVIDIA runtime
[`--runtime=nvidia`](https://github.com/NVIDIA/nvidia-docker#quick-start).

From here, you can follow the instructions for
[testing a development environment](#testing-the-development-environment).

### Building an optimized serving binary

When running TensorFlow Serving's ModelServer, you may notice a log message that
looks like this:

```console
I external/org_tensorflow/tensorflow/core/platform/cpu_feature_guard.cc:141]
Your CPU supports instructions that this TensorFlow binary was not compiled to
use: AVX2 FMA
```

This indicates that your ModelServer binary isn't fully optimized for the CPU
its running on. Depending on the model you are serving, further optimizations
may not be necessary. However, building an optimized binary is straight-forward.

When building a Docker image from the provided `Dockerfile.devel` or
`Dockerfile.devel-gpu` files, the ModelServer binary will be built with the flag
`-march=native`. This will cause Bazel to build a ModelServer binary with all of
the CPU optimizations the host you're building the Docker image on supports.

To create a serving image that's fully optimized for your host, simply:

1.  Clone the TensorFlow Serving project

    ```shell
    git clone https://github.com/tensorflow/serving
    cd serving
    ```

2.  Build an image with an optimized ModelServer

    *   For CPU:

        ```shell
        docker build --pull -t $USER/tensorflow-serving-devel \
          -f tensorflow_serving/tools/docker/Dockerfile.devel .
        ```

    *   For GPU: `

        ```shell
        docker build --pull -t $USER/tensorflow-serving-devel-gpu \
          -f tensorflow_serving/tools/docker/Dockerfile.devel-gpu .
        ```

3.  Build a serving image with the development image as a base

    *   For CPU:

        ```shell
        docker build -t $USER/tensorflow-serving \
          --build-arg TF_SERVING_BUILD_IMAGE=$USER/tensorflow-serving-devel \
          -f tensorflow_serving/tools/docker/Dockerfile .
        ```

        Your new optimized Docker image is now `$USER/tensorflow-serving`, which
        you can [use](#running-a-serving-image) just as you would the standard
        `tensorflow/serving:latest` image.

    *   For GPU:

        ```shell
        docker build -t $USER/tensorflow-serving-gpu \
          --build-arg TF_SERVING_BUILD_IMAGE=$USER/tensorflow-serving-devel-gpu \
          -f tensorflow_serving/tools/docker/Dockerfile.gpu .
        ```

        Your new optimized Docker image is now `$USER/tensorflow-serving-gpu`,
        which you can [use](#running-a-gpu-serving-image) just as you would the
        standard `tensorflow/serving:latest-gpu` image.
