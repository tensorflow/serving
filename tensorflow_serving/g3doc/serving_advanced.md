# Building Standard TensorFlow Model Server

This tutorial shows you how to use TensorFlow Serving components to build the
standard TensorFlow model server that dynamically discovers and serves new
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
  [mnist_export.py](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/example/mnist_export.py)
  that trains and exports multiple versions of the model.

  * A C++ file
  [main.cc](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/model_servers/main.cc)
  which is the standard TensorFlow model server that discovers new exported
  models and runs a [gRPC](http://www.grpc.io) service for serving them.

This tutorial steps through the following tasks:

  1. Train and export a TensorFlow model.
  2. Manage model versioning with TensorFlow Serving `ServerCore`.
  3. Configure batching using `SessionBundleSourceAdapterConfig`.
  4. Serve request with TensorFlow Serving `ServerCore`.
  5. Run and test the service.

Before getting started, please complete the [prerequisites](setup.md#prerequisites).

## Train And Export TensorFlow Model

Clear the export directory if it already exists:

~~~shell
$>rm -rf /tmp/mnist_model
~~~

Train (with 100 iterations) and export the first version of model:

~~~shell
$>bazel build //tensorflow_serving/example:mnist_export
$>bazel-bin/tensorflow_serving/example/mnist_export --training_iteration=100 --export_version=1 /tmp/mnist_model
~~~

Train (with 2000 iterations) and export the second version of model:

~~~shell
$>bazel-bin/tensorflow_serving/example/mnist_export --training_iteration=2000 --export_version=2 /tmp/mnist_model
~~~

As you can see in `mnist_export.py`, the training and exporting is done the same
way it is in the [TensorFlow Serving basic tutorial](serving_basic.md). For
demonstration purposes, you're intentionally dialing down the training
iterations for the first run and exporting it as v1, while training it normally
for the second run and exporting it as v2 to the same parent directory -- as we
expect the latter to achieve better classification accuracy due to more
intensive training. You should see training data for each training run in your
`mnist_model` directory:

~~~shell
$>ls /tmp/mnist_model
00000001  00000002
~~~

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

~~~c++
int main(int argc, char** argv) {
  ...

  std::unique_ptr<ServerCore> core;
  TF_CHECK_OK(ServerCore::Create(
      config, std::bind(CreateSourceAdapter, source_adapter_config,
                        std::placeholders::_1, std::placeholders::_2),
      &CreateServableStateMonitor, &LoadDynamicModelConfig, &core));
  RunServer(port, std::move(core));

  return 0;
}
~~~

`ServerCore::Create()` takes four parameters:

  * `ModelServerConfig` that specifies models to be loaded. Models are declared
  either through `model_config_list`, which declares a static list of models, or
  through `dynamic_model_config`, which declares a dynamic list of models that
  may get updated at runtime.
  * `SourceAdapterCreator` that creates the `SourceAdapter`, which adapts
  `StoragePath` (the path where a model version is discovered) to model
  `Loader` (loads the model version from storage path and provides state
  transition interfaces to the `Manager`). In this case, `CreateSourceAdapter`
  creates `SessionBundleSourceAdapter`, which we will explain later.
  * `ServableStateMonitorCreator` that creates `ServableStateMonitor`, which
  keeps track for `Servable` (model version) state transition and provides a
  query interface to the user. In this case, `CreateServableStateMonitor`
  creates the base `ServableStateMonitor`, which keeps track of servable states
  in memory. You can extend it to add state tracking capabilities (e.g. persists
  state change to disk, remote server, etc.)
  * `DynamicModelConfigLoader` that loads models from `dynamic_model_config`.
  The standard TensorFlow model server supports only `model_config_list` for
  now and therefore `LoadDynamicModelConfig` CHECK-fails when called. You can
  extend it to add dynamic model discovery/loading capabilities (e.g. through
  RPC, external service, etc.)

`SessionBundle` is a key component of TensorFlow Serving. It represents a
TensorFlow model loaded from a given path and provides the same `Session::Run`
interface as TensorFlow to run inference.
`SessionBundleSourceAdapter` adapts storage path to `Loader<SessionBundle>`
so that model lifetime can be managed by `Manager`.

With all these, `ServerCore` internally does the following:

  * Instantiates a `FileSystemStoragePathSource` that monitors model export
  paths declared in `model_config_list`.
  * Instantiates a `SourceAdapter` using the `SourceAdapterCreator` with the
  model type declared in `model_config_list` and connects the
  `FileSystemStoragePathSource` to it. This way, whenever a new model version is
  discovered under the export path, the `SessionBundleSourceAdapter` adapts it
  to a `Loader<SessionBundle>`.
  * Instantiates a specific implementation of `Manager` called
  `AspiredVersionsManager` that manages all such `Loader` instances created by
  the `SessionBundleSourceAdapter`.

Whenever a new version is available, this `AspiredVersionsManager` always
unloads the old version and replaces it with the new one. If you want to
start customizing, you are encouraged to understand the components that it
creates internally, and how to configure them.

It is worth mentioning that TensorFlow Serving is designed from scratch to be
very flexible and extensible. You can build various plugins to customize system
behavior, while taking advantage of generic core components like
`AspiredVersionsManager`. For example, you could build a data source plugin that
monitors cloud storage instead of local storage, or you could build a version
policy plugin that does version transition in a different way -- in fact, you
could even build a custom model plugin that serves non-TensorFlow models. These
topics are out of scope for this tutorial. However, you can refer to the
[custom source](custom_source.md) and [custom servable](custom_servable.md)
tutorials for more information.

## Batching

Another typical server feature we want in a production environment is batching.
Modern hardware accelerators (GPUs, etc.) used to do machine learning inference
usually achieve best computation efficiency when inference requests are run in
large batches.

Batching can be turned on by providing proper `SessionBundleSourceAdapterConfig`
when creating the `SessionBundleSourceAdapter`. In this case we set the
`BatchingParameters` with pretty much default values. Batching can be fine-tuned
by setting custom timeout, batch_size, etc. values. For details, please refer
to `BatchingParameters`.

~~~c++
SessionBundleSourceAdapterConfig source_adapter_config;
// Batching config
if (enable_batching) {
  BatchingParameters* batching_parameters =
      source_adapter_config.mutable_config()->mutable_batching_parameters();
  batching_parameters->mutable_thread_pool_name()->set_value(
      "model_server_batch_threads");
}
~~~

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
    * TensorFlow session or wrappers around them, such as `SessionBundle`.
    * Other kinds of machine-learned models.
    * Vocabulary lookup tables.
    * Embedding lookup tables.

  A composite model could be represented as multiple independent servables, or
  as a single composite servable. A servable may also correspond to a fraction
  of a Model, for example with a large lookup table sharded across many
  `Manager` instances.

To put all these into the context of this tutorial:

  * TensorFlow models are represented by one kind of servable --
  `SessionBundle`. `SessionBundle` internally consists of a `tensorflow:Session`
  paired with some metadata about what graph is loaded into the session and how
  to run it for inference.

  * There is a file-system directory containing a stream of TensorFlow exports,
  each in its own subdirectory whose name is a version number. The outer
  directory can be thought of as the serialized representation of the servable
  stream for the TensorFlow model being served. Each export corresponds to a
  servables that can be loaded.

  * `AspiredVersionsManager` monitors the export stream, and manages lifecycle
  of all SessionBundle` servables dynamically.

`TensorflowPredictImpl::Predict` then just:

  * Requests `SessionBundle` from the manager (through ServerCore).
  * Uses the `generic signatures` to map logical tensor names in `PredictRequest`
  to real tensor names and bind values to tensors.
  * Runs inference.

## Test and Run The Server

Copy the first version of the export to the monitored folder and start the
server.

~~~shell
$>mkdir /tmp/monitored
$>cp -r /tmp/mnist_model/00000001 /tmp/monitored
$>bazel build //tensorflow_serving/model_servers/tensorflow_model_server
$>bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --enable_batching --port=9000 --model_name=mnist --model_base_path=/tmp/monitored
~~~

The server will emit log messages every one second that say
"Aspiring version for servable ...", which means it has found the export, and is
tracking its continued existence.

Run the test with `--concurrency=10`. This will send concurrent requests to the
server and thus trigger your batching logic.

~~~shell
$>bazel build //tensorflow_serving/example:mnist_client
$>bazel-bin/tensorflow_serving/example/mnist_client --num_tests=1000 --server=localhost:9000 --concurrency=10
...
Inference error rate: 13.1%
~~~

Then we copy the second version of the export to the monitored folder and re-run
the test:

~~~shell
$>cp -r /tmp/mnist_model/00000002 /tmp/monitored
$>bazel-bin/tensorflow_serving/example/mnist_client --num_tests=1000 --server=localhost:9000 --concurrency=10
...
Inference error rate: 9.5%
~~~

This confirms that your server automatically discovers the new version and uses
it for serving!
