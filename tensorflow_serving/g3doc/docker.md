# Using TensorFlow Serving with Docker

One of the easiest ways to get started using TensorFlow Serving is with
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

See the Docker Hub
[tensorflow/serving repo](http://hub.docker.com/r/tensorflow/serving/tags/) for
other versions of images you can pull.

### Running a serving image

The serving images (both CPU and GPU) have the following properties:

*   Port 8500 exposed for gRPC
*   Port 8501 exposed for the REST API
*   Optional environment variable `MODEL_NAME` (defaults to `model`)
*   Optional environment variable `MODEL_BASE_PATH` (defaults to `/models`)

When the serving image runs ModelServer, it runs it as follows:

```shell
tensorflow_model_server --port=8500 --rest_api_port=8501 \
  --model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME}
```

To serve with Docker, you'll need:

*   An open port on your host to serve on
*   A SavedModel to serve
*   A name for your model that your client will refer to

What you'll do is
[run the Docker](https://docs.docker.com/engine/reference/run/) container,
[publish](https://docs.docker.com/engine/reference/commandline/run/#publish-or-expose-port--p---expose)
the container's ports to your host's ports, and mounting your host's path to the
SavedModel to where the container expects models.

Let's look at an example:

```shell
docker run -p 8501:8501 \
  --mount type=bind,source=/path/to/my_model/,target=/models/my_model \
  -e MODEL_NAME=my_model -t tensorflow/serving
```

In this case, we've started a Docker container, published the REST API port 8501
to our host's port 8501, and taken a model we named `my_model` and bound it to
the default model base path (`${MODEL_BASE_PATH}/${MODEL_NAME}` =
`/models/my_model`). Finally, we've filled in the environment variable
`MODEL_NAME` with `my_model`, and left `MODEL_BASE_PATH` to its default value.

This will run in the container:

```shell
tensorflow_model_server --port=8500 --rest_api_port=8501 \
  --model_name=my_model --model_base_path=/models/my_model
```

If we wanted to publish the gRPC port, we would use `-p 8500:8500`. You can have
both gRPC and REST API ports open at the same time, or choose to only open one
or the other.

#### Passing additional arguments

`tensorflow_model_server` supports many additional arguments that you could pass
to the serving docker containers. For example, if we wanted to pass a model
config file instead of specifying the model name, we could do the following:

```shell
docker run -p 8500:8500 -p 8501:8501 \
  --mount type=bind,source=/path/to/my_model/,target=/models/my_model \
  --mount type=bind,source=/path/to/my/models.config,target=/models/models.config \
  -t tensorflow/serving --model_config_file=/models/models.config
```

This approach works for any of the other command line arguments that
`tensorflow_model_server` supports.

### Creating your own serving image

If you want a serving image that has your model built into the container, you
can create your own image.

First run a serving image as a daemon:

```shell
docker run -d --name serving_base tensorflow/serving
```

Next, copy your SavedModel to the container's model folder:

```shell
docker cp models/<my model> serving_base:/models/<my model>
```

Finally, commit the container that's serving your model by changing `MODEL_NAME`
to match your model's name `<my model>':

```shell
docker commit --change "ENV MODEL_NAME <my model>" serving_base <my container>
```

You can now stop `serving_base`

```shell
docker kill serving_base
```

This will leave you with a Docker image called `<my container>` that you can
deploy and will load your model for serving on startup.

### Serving example

Let's run through a full example where we load a SavedModel and call it using
the REST API. First pull the serving image:

```shell
docker pull tensorflow/serving
```

This will pull the latest TensorFlow Serving image with ModelServer installed.

Next, we will use a toy model called `Half Plus Two`, which generates `0.5 * x +
2` for the values of `x` we provide for prediction.

To get this model, first clone the TensorFlow Serving repo.

```shell
mkdir -p /tmp/tfserving
cd /tmp/tfserving
git clone https://github.com/tensorflow/serving
```

Next, run the TensorFlow Serving container pointing it to this model and opening
the REST API port (8501):

```shell
docker run -p 8501:8501 \
  --mount type=bind,\
  source=/tmp/tfserving/serving/tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_two_cpu,\
  target=/models/half_plus_two \
  -e MODEL_NAME=half_plus_two -t tensorflow/serving &
```

This will run the docker container and launch the TensorFlow Serving Model
Server, bind the REST API port 8501, and map our desired model from our host to
where models are expected in the container. We also pass the name of the model
as an environment variable, which will be important when we query the model.

To query the model using the predict API, you can run

```shell
curl -d '{"instances": [1.0, 2.0, 5.0]}' \
  -X POST http://localhost:8501/v1/models/half_plus_two:predict
```

NOTE: Older versions of Windows and other systems without curl can download it
[here](https://curl.haxx.se/download.html).

This should return a set of values:

```json
{ "predictions": [2.5, 3.0, 4.5] }
```

More information on using the RESTful API can be found [here](api_rest.md).

## Serving with Docker using your GPU

### Install nvidia-docker

Before serving with a GPU, in addition to
[installing Docker](#installing_docker), you will need:

*   Up-to-date [NVIDIA drivers](http://www.nvidia.com/drivers) for your system
*   `nvidia-docker`: You can follow the
    [installation instructions here](https://github.com/NVIDIA/nvidia-docker#quick-start)

### Running a GPU serving image

Running a GPU serving image is identical to running a CPU image. For more
details, see [running a serving image](#running-a-serving-image).

### GPU Serving example

Let's run through a full example where we load a model with GPU-bound ops and
call it using the REST API.

First install [`nvidia-docker`](#install-nvidia-docker). Next you can pull the
latest TensorFlow Serving GPU docker image by running:

```shell
docker pull tensorflow/serving:latest-gpu
```

This will pull down an minimal Docker image with ModelServer built for running
on GPUs installed.

Next, we will use a toy model called `Half Plus Two`, which generates `0.5 * x +
2` for the values of `x` we provide for prediction. This model will have ops
bound to the GPU device, and will not run on the CPU.

To get this model, first clone the TensorFlow Serving repo.

```shell
mkdir -p /tmp/tfserving
cd /tmp/tfserving
git clone https://github.com/tensorflow/serving
```

Next, run the TensorFlow Serving container pointing it to this model and opening
the REST API port (8501):

```shell
docker run --runtime=nvidia -p 8501:8501 \
  --mount type=bind,\
  source=/tmp/tfserving/serving/tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_two_gpu,\
  target=/models/half_plus_two \
  -e MODEL_NAME=half_plus_two -t tensorflow/serving:latest-gpu &
```

This will run the docker container with the `nvidia-docker` runtime, launch the
TensorFlow Serving Model Server, bind the REST API port 8501, and map our
desired model from our host to where models are expected in the container. We
also pass the name of the model as an environment variable, which will be
important when we query the model.

TIP: Before querying the model, be sure to wait till you see a message like the
following, indicating that the server is ready to receive requests:

```shell
2018-07-27 00:07:20.773693: I tensorflow_serving/model_servers/main.cc:333]
Exporting HTTP/REST API at:localhost:8501 ...
```

To query the model using the predict API, you can run

```shell
$ curl -d '{"instances": [1.0, 2.0, 5.0]}' \
  -X POST http://localhost:8501/v1/models/half_plus_two:predict
```

NOTE: Older versions of Windows and other systems without curl can download it
[here](https://curl.haxx.se/download.html).

This should return a set of values:

```json
{ "predictions": [2.5, 3.0, 4.5] }
```

TIP: Trying to run the GPU model on a machine without a GPU or without a working
GPU build of TensorFlow Model Server will result in an error that looks like:

```shell
Cannot assign a device for operation 'a': Operation was explicitly assigned to /device:GPU:0
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

See the Docker Hub
[tensorflow/serving repo](http://hub.docker.com/r/tensorflow/serving/tags/) for
other versions of images you can pull.

### Development example

After pulling one of the development Docker images, you can run it while opening
the gRPC port (8500):

```shell
docker run -it -p 8500:8500 tensorflow/serving:latest-devel
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
you may limit RAM usage by specifying `--local_resources=2048,.5,1.0` while
invoking Bazel. See the
[Bazel docs](https://docs.bazel.build/versions/master/user-manual.html#flag--local_resources)
for more information. You can use this same mechanism to tweak the optmizations
you're building TensorFlow Serving with. For example:

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
