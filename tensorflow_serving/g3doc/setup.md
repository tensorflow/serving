# Installation

## Installing ModelServer

### Installing using Docker

The easiest and most straight-forward way of using TensorFlow Serving is with
[Docker images](docker.md). We highly recommend this route unless you have
specific needs that are not addressed by running in a container.

TIP: This is also the easiest way to get TensorFlow Serving working with [GPU
support](docker.md#serving-with-docker-using-your-gpu).

### Installing using APT

#### Available binaries

The TensorFlow Serving ModelServer binary is available in two variants:

**tensorflow-model-server**: Fully optimized server that uses some platform
specific compiler optimizations like SSE4 and AVX instructions. This should be
the preferred option for most users, but may not work on some older machines.

**tensorflow-model-server-universal**: Compiled with basic optimizations, but
doesn't include platform specific instruction sets, so should work on most if
not all machines out there. Use this if `tensorflow-model-server` does not work
for you. Note that the binary name is the same for both packages, so if you
already installed tensorflow-model-server, you should first uninstall it using

<!-- common_typos_disable -->

```shell
apt-get remove tensorflow-model-server
```

<!-- common_typos_enable -->

#### Installation

1.  Add TensorFlow Serving distribution URI as a package source (one time setup)

    ```shell
    echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list && \
    curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -
    ```

2.  Install and update TensorFlow ModelServer

    <!-- common_typos_disable -->

    ```shell
    apt-get update && apt-get install tensorflow-model-server
    ```

    <!-- common_typos_enable -->

Once installed, the binary can be invoked using the command
    `tensorflow_model_server`.


You can upgrade to a newer version of tensorflow-model-server with:

<!-- common_typos_disable -->

```shell
apt-get upgrade tensorflow-model-server
```

<!-- common_typos_enable -->

Note: In the above commands, replace tensorflow-model-server with
tensorflow-model-server-universal if your processor does not support AVX
instructions.

## Building from source

The recommended approach to building from source is to use Docker. The
TensorFlow Serving Docker development images encapsulate all the dependencies
you need to build your own version of TensorFlow Serving.

For a listing of what these dependencies are, see the TensorFlow Serving
Development Dockerfiles
[[CPU](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/tools/docker/Dockerfile.devel),
[GPU](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/tools/docker/Dockerfile.devel-gpu)].

Note: Currently we only support building binaries that run on Linux.

#### Installing Docker

General installation instructions are
[on the Docker site](https://docs.docker.com/install/).

#### Clone the build script

After installing Docker, we need to get the source we want to build from. We
will use Git to clone the master branch of TensorFlow Serving:

```shell
git clone https://github.com/tensorflow/serving.git
cd serving
```

#### Build

In order to build in a hermetic environment with all dependencies taken care of,
we will use the `run_in_docker.sh` script. This script passes build commands
through to a Docker container. By default, the script will build with the latest
nightly Docker development image.

TensorFlow Serving uses Bazel as its build tool. You can use Bazel commands to
build individual targets or the entire source tree.

To build the entire tree, execute:

```shell
tools/run_in_docker.sh bazel build -c opt tensorflow_serving/...
```

Binaries are placed in the bazel-bin directory, and can be run using a command
like:

```shell
bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server
```

To test your build, execute:

```shell
tools/run_in_docker.sh bazel test -c opt tensorflow_serving/...
```

See the [basic tutorial](serving_basic.md) and [advanced
tutorial](serving_advanced.md) for more in-depth examples of running TensorFlow
Serving.

##### Building specific versions of TensorFlow Serving

If you want to build from a specific branch (such as a release branch), pass `-b
<branchname>` to the `git clone` command.

We will also want to match the build environment for that branch of code, by
passing the `run_in_docker.sh` script the Docker development image we'd like to
use.

For example, to build version 1.10 of TensorFlow Serving:

```console
$ git clone -b r1.10 https://github.com/tensorflow/serving.git
...
$ cd serving
$ tools/run_in_docker.sh -d tensorflow/serving:1.10-devel \
  bazel build tensorflow_serving/...
...
```

##### Optimized build

If you'd like to apply generally recommended optimizations, including utilizing
platform-specific instruction sets for your processor, you can add
`--config=nativeopt` to Bazel build commands when building TensorFlow Serving.

For example:

```shell
tools/run_in_docker.sh bazel build --config=nativeopt tensorflow_serving/...
```

It's also possible to compile using specific instruction sets (e.g. AVX).
Wherever you see `bazel build` in the documentation, simply add the
corresponding flags:

Instruction Set            | Flags
-------------------------- | ----------------------
AVX                        | `--copt=-mavx`
AVX2                       | `--copt=-mavx2`
FMA                        | `--copt=-mfma`
SSE 4.1                    | `--copt=-msse4.1`
SSE 4.2                    | `--copt=-msse4.2`
All supported by processor | `--copt=-march=native`

For example:

```shell
tools/run_in_docker.sh bazel build --copt=-mavx2 tensorflow_serving/...
```

Note: These instruction sets are not available on all machines, especially with
older processors. Use the default `--config=nativeopt` to build an optimized
version of TensorFlow Serving for your processor if you are in doubt.


##### Building with GPU Support

In order to build a custom version of TensorFlow Serving with GPU support, we
recommend either building with the
[provided Docker images](building_with_docker.md), or following the approach in
the
[GPU Dockerfile](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/tools/docker/Dockerfile.devel-gpu).

## TensorFlow Serving Python API PIP package

To run Python client code without the need to build the API, you can install the
`tensorflow-serving-api` PIP package using:

```shell
pip install tensorflow-serving-api
```
