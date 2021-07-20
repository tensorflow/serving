# TensorFlow Serving with Docker

One of the easiest ways to get started using TensorFlow Serving is with
[Docker](http://www.docker.com/).

<pre class="prettyprint lang-bsh">
# Download the TensorFlow Serving Docker image and repo
<code class="devsite-terminal">docker pull tensorflow/serving</code><br/>
<code class="devsite-terminal">git clone https://github.com/tensorflow/serving</code>
# Location of demo models
<code class="devsite-terminal">TESTDATA="$(pwd)/serving/tensorflow_serving/servables/tensorflow/testdata"</code>

# Start TensorFlow Serving container and open the REST API port
<code class="devsite-terminal">docker run -t --rm -p 8501:8501 \
    -v "$TESTDATA/saved_model_half_plus_two_cpu:/models/half_plus_two" \
    -e MODEL_NAME=half_plus_two \
    tensorflow/serving &</code>

# Query the model using the predict API
<code class="devsite-terminal">curl -d '{"instances": [1.0, 2.0, 5.0]}' \
    -X POST http://localhost:8501/v1/models/half_plus_two:predict</code><br/>
# Returns => { "predictions": [2.5, 3.0, 4.5] }
</pre>

For additional serving endpoints, see the <a href="./api_rest.md">Client REST API</a>.

## Install Docker

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

This will pull down a minimal Docker image with TensorFlow Serving installed.

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

First install [`nvidia-docker`](https://github.com/NVIDIA/nvidia-docker#quick-start). Next you can pull the
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
docker run --gpus all -p 8501:8501 \
--mount type=bind,\
source=/tmp/tfserving/serving/tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_two_gpu,\
target=/models/half_plus_two \
  -e MODEL_NAME=half_plus_two -t tensorflow/serving:latest-gpu &
```

This will run the docker container, launch the
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
curl -d '{"instances": [1.0, 2.0, 5.0]}' \
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

For instructions on how to build and develop Tensorflow Serving, please refer to
[Developing with Docker guide](building_with_docker.md).
