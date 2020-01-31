# Building Standard TensorFlow ModelServer

This tutorial shows you how to use TensorFlow Serving components to build the
standard TensorFlow ModelServer that dynamically discovers and serves new
versions of a trained TensorFlow model. If you just want to use the standard
server to serve your models, see
[TensorFlow Serving basic tutorial](serving_basic.md).

This tutorial uses the simple Softmax Regression model introduced in the
TensorFlow tutorial for handwritten image (MNIST data) classification. If you
don't know what TensorFlow or MNIST is, see the
[MNIST For ML Beginners](http://www.tensorflow.org/tutorials/mnist/beginners/index.html#mnist-for-ml-beginners)
tutorial.

The code for this tutorial consists of two parts:

  * A Python file
  [mnist_saved_model.py](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/example/mnist_saved_model.py)
  that trains and exports multiple versions of the model.

  * A C++ file
  [main.cc](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/model_servers/main.cc)
  which is the standard TensorFlow ModelServer that discovers new exported
  models and runs a [gRPC](http://www.grpc.io) service for serving them.

This tutorial steps through the following tasks:

1.  Train and export a TensorFlow model.
2.  Manage model versioning with TensorFlow Serving `ServerCore`.
3.  Configure batching using `SavedModelBundleSourceAdapterConfig`.
4.  Serve request with TensorFlow Serving `ServerCore`.
5.  Run and test the service.

Before getting started, first [install Docker](docker.md#installing-docker)

## Train and export TensorFlow Model

First, if you haven't done so yet, clone this repository to your local machine:

```shell
git clone https://github.com/tensorflow/serving.git
cd serving
```

Clear the export directory if it already exists:

```shell
rm -rf /tmp/models
```

Train (with 100 iterations) and export the first version of model:

```shell
tools/run_in_docker.sh python tensorflow_serving/example/mnist_saved_model.py \
  --training_iteration=100 --model_version=1 /tmp/mnist
```

Train (with 2000 iterations) and export the second version of model:

```shell
tools/run_in_docker.sh python tensorflow_serving/example/mnist_saved_model.py \
  --training_iteration=2000 --model_version=2 /tmp/mnist
```

As you can see in `mnist_saved_model.py`, the training and exporting is done the
same way it is in the [TensorFlow Serving basic tutorial](serving_basic.md). For
demonstration purposes, you're intentionally dialing down the training
iterations for the first run and exporting it as v1, while training it normally
for the second run and exporting it as v2 to the same parent directory -- as we
expect the latter to achieve better classification accuracy due to more
intensive training. You should see training data for each training run in your
`/tmp/mnist` directory:

```console
$ ls /tmp/mnist
1  2
```

## ServerCore

Now imagine v1 and v2 of the model are dynamically generated at runtime, as new
algorithms are being experimented with, or as the model is trained with a new
data set. In a production environment, you may want to build a server that can
support gradual rollout, in which v2 can be discovered, loaded, experimented,
monitored, or reverted while serving v1. Alternatively, you may want to tear
down v1 before bringing up v2. TensorFlow Serving supports both options -- while
one is good for maintaining availability during the transition, the other is
good for minimizing resource usage (e.g. RAM).

TensorFlow Serving `Manager` does exactly that. It handles the full lifecycle of
TensorFlow models including loading, serving and unloading them as well as
version transitions. In this tutorial, you will build your server on top of a
TensorFlow Serving `ServerCore`, which internally wraps an
`AspiredVersionsManager`.

```c++
int main(int argc, char** argv) {
  ...

  ServerCore::Options options;
  options.model_server_config = model_server_config;
  options.servable_state_monitor_creator = &CreateServableStateMonitor;
  options.custom_model_config_loader = &LoadCustomModelConfig;

  ::google::protobuf::Any source_adapter_config;
  SavedModelBundleSourceAdapterConfig
      saved_model_bundle_source_adapter_config;
  source_adapter_config.PackFrom(saved_model_bundle_source_adapter_config);
  (*(*options.platform_config_map.mutable_platform_configs())
      [kTensorFlowModelPlatform].mutable_source_adapter_config()) =
      source_adapter_config;

  std::unique_ptr<ServerCore> core;
  TF_CHECK_OK(ServerCore::Create(options, &core));
  RunServer(port, std::move(core));

  return 0;
}
```

`ServerCore::Create()` takes a ServerCore::Options parameter. Here are a few
commonly used options:

  * `ModelServerConfig` that specifies models to be loaded. Models are declared
  either through `model_config_list`, which declares a static list of models, or
  through `custom_model_config`, which defines a custom way to declare a list of
  models that may get updated at runtime.
  * `PlatformConfigMap` that maps from the name of the platform (such as
  `tensorflow`) to the `PlatformConfig`, which is used to create the
  `SourceAdapter`. `SourceAdapter` adapts `StoragePath` (the path where a model
  version is discovered) to model `Loader` (loads the model version from
  storage path and provides state transition interfaces to the `Manager`). If
  `PlatformConfig` contains `SavedModelBundleSourceAdapterConfig`, a
  `SavedModelBundleSourceAdapter` will be created, which we will explain later.

`SavedModelBundle` is a key component of TensorFlow Serving. It represents a
TensorFlow model loaded from a given path and provides the same `Session::Run`
interface as TensorFlow to run inference. `SavedModelBundleSourceAdapter` adapts
storage path to `Loader<SavedModelBundle>` so that model lifetime can be managed
by `Manager`. Please note that `SavedModelBundle` is the successor of deprecated
`SessionBundle`. Users are encouraged to use `SavedModelBundle` as the support
for `SessionBundle` will soon be removed.

With all these, `ServerCore` internally does the following:

  * Instantiates a `FileSystemStoragePathSource` that monitors model export
  paths declared in `model_config_list`.
  * Instantiates a `SourceAdapter` using the `PlatformConfigMap` with the
  model platform declared in `model_config_list` and connects the
  `FileSystemStoragePathSource` to it. This way, whenever a new model version is
  discovered under the export path, the `SavedModelBundleSourceAdapter`
  adapts it to a `Loader<SavedModelBundle>`.
  * Instantiates a specific implementation of `Manager` called
  `AspiredVersionsManager` that manages all such `Loader` instances created by
  the `SavedModelBundleSourceAdapter`. `ServerCore` exports the `Manager`
  interface by delegating the calls to `AspiredVersionsManager`.

Whenever a new version is available, this `AspiredVersionsManager` loads the new
version, and under its default behavior unloads the old one. If you want to
start customizing, you are encouraged to understand the components that it
creates internally, and how to configure them.

It is worth mentioning that TensorFlow Serving is designed from scratch to be
very flexible and extensible. You can build various plugins to customize system
behavior, while taking advantage of generic core components like `ServerCore`
and `AspiredVersionsManager`. For example, you could build a data source plugin
that monitors cloud storage instead of local storage, or you could build a
version policy plugin that does version transition in a different way -- in
fact, you could even build a custom model plugin that serves non-TensorFlow
models. These topics are out of scope for this tutorial. However, you can refer
to the [custom source](custom_source.md) and
[custom servable](custom_servable.md) tutorials for more information.

## Batching

Another typical server feature we want in a production environment is batching.
Modern hardware accelerators (GPUs, etc.) used to do machine learning inference
usually achieve best computation efficiency when inference requests are run in
large batches.

Batching can be turned on by providing proper `SessionBundleConfig` when
creating the `SavedModelBundleSourceAdapter`. In this case we set the
`BatchingParameters` with pretty much default values. Batching can be fine-tuned
by setting custom timeout, batch_size, etc. values. For details, please refer
to `BatchingParameters`.

```c++
SessionBundleConfig session_bundle_config;
// Batching config
if (enable_batching) {
  BatchingParameters* batching_parameters =
      session_bundle_config.mutable_batching_parameters();
  batching_parameters->mutable_thread_pool_name()->set_value(
      "model_server_batch_threads");
}
*saved_model_bundle_source_adapter_config.mutable_legacy_config() =
    session_bundle_config;
```

Upon reaching full batch, inference requests are merged internally into a
single large request (tensor), and `tensorflow::Session::Run()` is invoked
(which is where the actual efficiency gain on GPUs comes from).


# Serve with Manager

As mentioned above, TensorFlow Serving `Manager` is designed to be a generic
component that can handle loading, serving, unloading and version transition of
models generated by arbitrary machine learning systems. Its APIs are built
around the following key concepts:

  * **Servable**:
  Servable is any opaque object that can be used to serve client requests. The
  size and granularity of a servable is flexible, such that a single servable
  might include anything from a single shard of a lookup table to a single
  machine-learned model to a tuple of models. A servable can be of any type and
  interface.

  * **Servable Version**:
  Servables are versioned and TensorFlow Serving `Manager` can manage one or
  more versions of a servable. Versioning allows for more than one version of a
  servable to be loaded concurrently, supporting gradual rollout and
  experimentation.

  * **Servable Stream**:
  A servable stream is the sequence of versions of a servable, with increasing
  version numbers.

  * **Model**:
  A machine-learned model is represented by one or more servables. Examples of
  servables are:
    * TensorFlow session or wrappers around them, such as `SavedModelBundle`.
    * Other kinds of machine-learned models.
    * Vocabulary lookup tables.
    * Embedding lookup tables.

    A composite model could be represented as multiple independent servables, or
    as a single composite servable. A servable may also correspond to a fraction
    of a Model, for example with a large lookup table sharded across many
    `Manager` instances.

To put all these into the context of this tutorial:

  * TensorFlow models are represented by one kind of servable --
    `SavedModelBundle`. `SavedModelBundle` internally consists of a
    `tensorflow:Session` paired with some metadata about what graph is loaded
    into the session and how to run it for inference.

  * There is a file-system directory containing a stream of TensorFlow exports,
    each in its own subdirectory whose name is a version number. The outer
    directory can be thought of as the serialized representation of the servable
    stream for the TensorFlow model being served. Each export corresponds to a
    servables that can be loaded.

  * `AspiredVersionsManager` monitors the export stream, and manages lifecycle
    of all `SavedModelBundle` servables dynamically.

`TensorflowPredictImpl::Predict` then just:

  * Requests `SavedModelBundle` from the manager (through ServerCore).
  * Uses the `generic signatures` to map logical tensor names in
  `PredictRequest` to real tensor names and bind values to tensors.
  * Runs inference.

## Test and run the server

Copy the first version of the export to the monitored folder:

```shell
mkdir /tmp/monitored
cp -r /tmp/mnist/1 /tmp/monitored
```

Then start the server:

```shell
docker run -p 8500:8500 \
  --mount type=bind,source=/tmp/monitored,target=/models/mnist \
  -t --entrypoint=tensorflow_model_server tensorflow/serving --enable_batching \
  --port=8500 --model_name=mnist --model_base_path=/models/mnist &
```

The server will emit log messages every one second that say
"Aspiring version for servable ...", which means it has found the export, and is
tracking its continued existence.

Let's run the client with `--concurrency=10`. This will send concurrent requests
to the server and thus trigger your batching logic.

```shell
tools/run_in_docker.sh python tensorflow_serving/example/mnist_client.py \
  --num_tests=1000 --server=127.0.0.1:8500 --concurrency=10
```

Which results in output that looks like:

```console
...
Inference error rate: 13.1%
```

Then we copy the second version of the export to the monitored folder and re-run
the test:

```shell
cp -r /tmp/mnist/2 /tmp/monitored
tools/run_in_docker.sh python tensorflow_serving/example/mnist_client.py \
  --num_tests=1000 --server=127.0.0.1:8500 --concurrency=10
```

Which results in output that looks like:

```console
...
Inference error rate: 9.5%
```

This confirms that your server automatically discovers the new version and uses
it for serving!
