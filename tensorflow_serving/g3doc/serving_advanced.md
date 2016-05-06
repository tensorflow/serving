# Serving Dynamically Updated TensorFlow Model with Asynchronous Batching

This tutorial shows you how to use TensorFlow Serving components to build a
server that dynamically discovers and serves new versions of a trained
TensorFlow model. You'll also learn how to use TensorFlow Serving's more
flexible, lower-level batch scheduling API. One advantage of the lower-level API
is its asynchronous behavior, which allows you to reduce the number of client
threads and thus use less memory without compromising throughput. The code
examples in this tutorial focus on the discovery, asynchronous batching, and
serving logic. If you just want to use TensorFlow Serving to serve a single
version model, and are fine with synchronous batching (relying on many client
threads that block), see [TensorFlow Serving basic tutorial](serving_basic.md).

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
  [mnist_inference_2.cc](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/example/mnist_inference_2.cc)
  that discovers new exported models and runs a [gRPC](http://www.grpc.io)
  service for serving them.

This tutorial steps through the following tasks:

  1. Train and export a TensorFlow model.
  2. Manage model versioning with TensorFlow Serving manager.
  3. Handle request batching with TensorFlow Serving batch scheduler.
  4. Serve request with TensorFlow Serving manager.
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

## Manager

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
TensorFlow Serving `Manager` (see `mnist_inference_2.cc`).

~~~c++
int main(int argc, char** argv) {
  ...

  UniquePtrWithDeps<tensorflow::serving::Manager> manager;
  tensorflow::Status status = tensorflow::serving::simple_servers::
      CreateSingleTFModelManagerFromBasePath(export_base_path, &manager);

  ...

  RunServer(FLAGS_port, ready_ids[0].name, std::move(manager));

  return 0;
}
~~~

`CreateSingleTFModelManagerFromBasePath()` internally does the following:

  * Instantiate a `FileSystemStoragePathSource` that monitors new files (model
  export directory) in `export_base_path`.
  * Instantiate a `SessionBundleSourceAdapter` that creates a new
  `SessionBundle` for each new model export.
  * Instantiate a specific implementation of `Manager` called
  `AspiredVersionsManager` that manages all `SessionBundle` instances created by
  the `SessionBundleSourceAdapter`.

Whenever a new version is available, this `AspiredVersionsManager` always
unloads the old version and replaces it with the new one. Note that
`CreateSingleTFModelManagerFromBasePath()` intentionally lacks any config
parameters, because it is intended for a very first deployment. If you want to
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

TensorFlow Serving `BatchScheduler` provides such functionality. A specific
implementation of it -- `BasicBatchScheduler` -- enqueues tasks (requests)
until either of the following occur:

  * The next task would cause the batch to exceed the size target.
  * Waiting for more tasks to be added would exceed the timeout.

When either of these occur, the `BasicBatchScheduler` processes the entire
batch by executing a callback and passing it the current batch of tasks.

Initializing `BasicBatchScheduler` is straightforward and is done in a
`MnistServiceImpl` constructor. A `BasicBatchScheduler::Options` is given to
configure the batch scheduler, and a callback is given to be executed when the
batch is full or timeout exceeds.

The default `BasicBatchScheduler::Options` has batch size set to 32 and
timeout set to 10 milliseconds. These parameters are typically extremely
performance critical and should be tuned based on a specific model/scenario in
production. In this tutorial, you will not bother tuning them.

For each incoming request, instead of immediately processing it, you always
submit a task, which encapsulates gRPC async call data, to `BatchScheduler`:

~~~c++
void MnistServiceImpl::Classify(CallData* calldata) {
  ...

  std::unique_ptr<Task> task(new Task(calldata));
  tensorflow::Status status = batch_scheduler_->Schedule(std::move(task));
}
~~~

Upon reaching timeout or full batch, the given callback `DoClassifyInBatch()`
will be executed. `DoClassifyInBatch()`, as we will explain later, merges tasks
into a single large tensor, and invokes a single `tensorflow::Session::Run()`
call (which is where the actual efficiency gain on GPUs comes from).


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

`DoClassifyInBatch` then just requests `SessionBundle` from the manager and uses
it to run inference. Most of the logic and flow is very similar to the logic and
flow described in the [TensorFlow Serving basic tutorial](serving_basic.md),
with just a few key changes:

  * The input tensor now has its first dimension set to variable batch size at
  runtime, because you are batching multiple inference requests in a single
  input.

~~~c++
  Tensor input(tensorflow::DT_FLOAT, {batch_size, kImageDataSize});
~~~

  * You are calling `GetServableHandle` to request the `SessionBundle` of a
  version of the model. With `ServableRequest::Latest()` you create a
  `ServableRequest` that tells the `Manager` to return the latest version of the
  servable for a given name. You can also specify a version. Note that
  `SessionBundle` is wrapped in a `ServableHandle`. This is due to the fact
  that the lifetime of `SessionBundle` is now managed by the `Manager`,
  therefore a handle/reference is returned instead.

~~~c++
  auto handle_request =
      tensorflow::serving::ServableRequest::Latest(servable_name_);
  tensorflow::serving::ServableHandle<tensorflow::serving::SessionBundle> bundle;
  const tensorflow::Status lookup_status =
      manager_->GetServableHandle(handle_request, &bundle);
~~~

## Test and Run The Server

Copy the first version of the export to the monitored folder and start the
server.

~~~shell
$>mkdir /tmp/monitored
$>cp -r /tmp/mnist_model/00000001 /tmp/monitored
$>bazel build //tensorflow_serving/example:mnist_inference_2
$>bazel-bin/tensorflow_serving/example/mnist_inference_2 --port=9000 /tmp/monitored
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
