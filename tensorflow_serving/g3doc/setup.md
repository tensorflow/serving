# Installation

## Prerequisites

To compile and use TensorFlow Serving, you need to set up some prerequisites.

### Bazel (only if compiling source code)

TensorFlow Serving requires Bazel 0.5.4 or higher. You can find the Bazel
installation instructions [here](http://bazel.build/docs/install.html).

If you have the prerequisites for Bazel, those instructions consist of the
following steps:

1.  Download the relevant binary from
    [here](https://github.com/bazelbuild/bazel/releases). Let's say you
    downloaded bazel-0.5.4-installer-linux-x86_64.sh. You would execute:

    <pre>
    cd ~/Downloads
    chmod +x bazel-0.5.4-installer-linux-x86_64.sh
    ./bazel-0.5.4-installer-linux-x86_64.sh --user
    </pre>

2.  Set up your environment. Put this in your ~/.bashrc.

    <pre>
    export PATH="$PATH:$HOME/bin"
    </pre>

### gRPC

Our tutorials use [gRPC](http://www.grpc.io) (1.0.0 or higher) as our RPC
framework. You can find the installation instructions
[here](https://github.com/grpc/grpc/tree/master/src/python/grpcio).

### Packages

To install TensorFlow Serving dependencies, execute the following:

```shell
sudo apt-get update && sudo apt-get install -y \
        build-essential \
        curl \
        libcurl3-dev \
        git \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python-dev \
        python-numpy \
        python-pip \
        software-properties-common \
        swig \
        zip \
        zlib1g-dev
```

The list of packages needed to build TensorFlow changes over time, so if you
encounter any issues, refer TensorFlow's [build
instructions](https://www.tensorflow.org/install/install_sources). Pay
particular attention to `apt-get install` and `pip install` commands which you
may need to run.

### TensorFlow Serving Python API PIP package

To run Python client code without the need to install Bazel, you can install
the `tensorflow-serving-api` PIP package using:

```shell
pip install tensorflow-serving-api
```

## Installing using apt-get

### Available binaries

The TensorFlow Serving ModelServer binary is available in two variants:

**tensorflow-model-server**: Fully optimized server that uses some platform
specific compiler optimizations like SSE4 and AVX instructions. This should be
the preferred option for most users, but may not work on some older machines.

**tensorflow-model-server-universal**: Compiled with basic optimizations, but
doesn't include platform specific instruction sets, so should work on most if
not all machines out there. Use this if `tensorflow-model-server` does not work
for you. Note that the binary name is the same for both packages, so if you
already installed tensorflow-model-server, you should first uninstall it using

```shell
sudo apt-get remove tensorflow-model-server
```

### Installing the ModelServer

1.  Add TensorFlow Serving distribution URI as a package source (one time setup)

    <pre>
    echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list

    curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -
    </pre>

2.  Install and update TensorFlow ModelServer

    <pre>
    sudo apt-get update && sudo apt-get install tensorflow-model-server
    </pre>

Once installed, the binary can be invoked using the command `tensorflow_model_server`.

You can upgrade to a newer version of tensorflow-model-server with:

```shell
sudo apt-get upgrade tensorflow-model-server
```

Note: In the above commands, replace tensorflow-model-server with
tensorflow-model-server-universal if your processor does not support AVX
instructions.

## Installing from source

### Clone the TensorFlow Serving repository

```shell
git clone --recurse-submodules https://github.com/tensorflow/serving
cd serving
```

`--recurse-submodules` is required to fetch TensorFlow, gRPC, and other
libraries that TensorFlow Serving depends on. Note that these instructions
will install the latest master branch of TensorFlow Serving. If you want to
install a specific branch (such as a release branch), pass `-b <branchname>`
to the `git clone` command.

### Install prerequisites

Follow the Prerequisites section above to install all dependencies.
To configure TensorFlow, run

```shell
cd tensorflow
./configure
cd ..
```

Consult the
[TensorFlow install instructions](https://www.tensorflow.org/install/)
if you encounter any issues with setting up TensorFlow or its dependencies.


### Build

TensorFlow Serving uses Bazel to build. Use Bazel commands to build individual
targets or the entire source tree.

To build the entire tree, execute:

```shell
bazel build -c opt tensorflow_serving/...
```

Binaries are placed in the bazel-bin directory, and can be run using a command
like:

```shell
bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server
```

To test your installation, execute:

```shell
bazel test -c opt tensorflow_serving/...
```

See the [basic tutorial](serving_basic.md) and [advanced tutorial](serving_advanced.md)
for more in-depth examples of running TensorFlow Serving.

### Optimized build

It's possible to compile using some platform specific instruction sets (e.g.
AVX) that can significantly improve performance. Wherever you see 'bazel build'
in the documentation, you can add the flags `-c opt --copt=-msse4.1
--copt=-msse4.2 --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-O3` (or some
subset of these flags). For example:

```shell
bazel build -c opt --copt=-msse4.1 --copt=-msse4.2 --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-O3 tensorflow_serving/...
```

Note: These instruction sets are not available on all machines, especially with
older processors, so it may not work with all flags. You can try some subset of
them, or revert to just the basic '-c opt' which is guaranteed to work on all
machines.

### Continuous integration build

Our [continuous integration build](http://ci.tensorflow.org/view/Serving/job/serving-master-cpu/)
using TensorFlow [ci_build](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/ci_build)
infrastructure offers you simplified development using docker. All you need is
git and docker. No need to install all other dependencies manually.

```shell
git clone --recursive https://github.com/tensorflow/serving
cd serving
CI_TENSORFLOW_SUBMODULE_PATH=tensorflow tensorflow/tensorflow/tools/ci_build/ci_build.sh CPU bazel test //tensorflow_serving/...
```

Note: The `serving` directory is mapped into the container. You can develop
outside the docker container (in your favourite editor) and when you run this
build it will build with your changes.
