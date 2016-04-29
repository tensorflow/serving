# Serving a TensorFlow Model

This tutorial shows you how to use TensorFlow Serving components to export a
trained TensorFlow model and build a server to serve the exported model. The
server you will build in this tutorial is relatively simple: it serves a single
static TensorFlow model, it handles inference requests, and it calculates an
aggregate inference error rate. If you are already familiar with TensorFlow
Serving, and you want to create a more sophisticated server that handles
inference requests asynchronously (without blocking client threads), and
discovers and serves new versions of a TensorFlow model that is being
dynamically updated, see the
[TensorFlow Serving advanced tutorial](serving_advanced.md).

This tutorial uses the simple Softmax Regression model introduced in the
TensorFlow tutorial for handwritten image (MNIST data) classification. If you
do not know what TensorFlow or MNIST is, see the
[MNIST For ML Beginners](http://www.tensorflow.org/tutorials/mnist/beginners/index.html#mnist-for-ml-beginners)
tutorial.

The code for this tutorial consists of two parts:

* A Python file
([mnist_export.py](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/example/mnist_export.py))
that trains and exports the model.

* A C++ file
([mnist_inference.cc](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/example/mnist_inference.cc))
that loads the exported model and runs a [gRPC](http://www.grpc.io) service to
serve it.

Before getting started, please complete the [prerequisites](setup.md#prerequisites).

## Train And Export TensorFlow Model

As you can see in `mnist_export.py`, the training is done the same way it is in
the MNIST For ML Beginners tutorial. The TensorFlow graph is launched in
TensorFlow session `sess`, with the input tensor (image) as `x` and output
tensor (Softmax score) as `y`.

Then we use TensorFlow Serving `Exporter` module to export the model.
`Exporter` saves a "snapshot" of the trained model to reliable storage so that
it can be loaded later for inference.

~~~python
from tensorflow_serving.session_bundle import exporter
...
export_path = sys.argv[-1]
print 'Exporting trained model to', export_path
saver = tf.train.Saver(sharded=True)
model_exporter = exporter.Exporter(saver)
signature = exporter.classification_signature(input_tensor=x, scores_tensor=y)
model_exporter.init(sess.graph.as_graph_def(),
                    default_graph_signature=signature)
model_exporter.export(export_path, tf.constant(FLAGS.export_version), sess)
~~~

`Exporter.__init__` takes a `tensorflow.train.Saver`, with the only requirement
being that `Saver` should have `sharded=True`. `saver` is used to serialize
graph variable values to the model export so that they can be properly restored
later. Note that since no `variable_list` is specified for the `Saver`, it will
export all variables of the graph. For more complex graphs, you can choose to
export only the variables that will be used for inference.

`Exporter.init()` takes the following arguments:

  * `sess.graph.as_graph_def()` is the
  [protobuf](https://developers.google.com/protocol-buffers/) of the graph.
  `export` will serialize the protobuf to the model export so that the
  TensorFlow graph can be properly restored later.

  * `default_graph_signature=signature` specifies a model export **signature**.
  Signature specifies what type of model is being exported, and the input/output
  tensors to bind to when running inference. In this case, you use
  `exporter.classification_signature` to specify that the model is a
  classification model:

    * `input_tensor=x` specifies the input tensor binding.

    * `scores_tensor=y` specifies the scores tensor binding.

    * Typically, you should also override the `classes_tensor=None` argument to
    specify class tensor binding. As an example, for a classification model that
    returns the top 10 suggested videos, the output will consist of both videos
    (classes) and scores for each video. In this case, however, the model always
    returns Softmax scores for matching digit 0-9 in order. Therefore,
    overriding `classes_tensor` is not necessary.

`Exporter.export()` takes the following arguments:

  * `export_path` is the path of the export directory. `export` will create the
  directory if it does not exist.

  * `tf.constant(FLAGS.export_version)` is a tensor that specifies the
  **version** of the model. You should specify a larger integer value when
  exporting a newer version of the same model. Each version will be exported to
  a different sub-directory under the given path.

  * `sess` is the TensorFlow session that holds the trained model you are
  exporting.

Let's run it!

Clear the export directory if it already exists:

~~~shell
$>rm -rf /tmp/mnist_model
~~~

~~~shell
$>bazel build //tensorflow_serving/example:mnist_export
$>bazel-bin/tensorflow_serving/example/mnist_export /tmp/mnist_model
Training model...

...

Done training!
Exporting trained model to /tmp/mnist_model
Done exporting!
~~~

Now let's take a look at the export directory.

~~~shell
$>ls /tmp/mnist_model
00000001
~~~

As mentioned above, a sub-directory will be created for exporting each version
of the model. You specified `tf.constant(FLAGS.export_version)` as the model
version above, and `FLAGS.export_version` has the default value of 1, therefore
the corresponding sub-directory `00000001` is created.

~~~shell
$>ls /tmp/mnist_model/00000001
checkpoint export-00000-of-00001 export.meta
~~~

Each version sub-directory contains the following files:

  * `export.meta` is the serialized tensorflow::MetaGraphDef of the model. It
  includes the graph definition of the model, as well as metadata of the model
  such as signatures.

  * `export-?????-of-?????` are files that hold the serialized variables of
  the graph.

With that, your TensorFlow model is exported and ready to be loaded!

## Load Exported TensorFlow Model

The C++ code for loading the exported TensorFlow model is in the `main()`
function in mnist_inference.cc, and is simple. The basic code is:

~~~c++
int main(int argc, char** argv) {
  ...

  SessionBundleConfig session_bundle_config;
  ... (ignoring batching for now; see below)
  std::unique_ptr<SessionBundleFactory> bundle_factory;
  TF_QCHECK_OK(
      SessionBundleFactory::Create(session_bundle_config, &bundle_factory));
  std::unique_ptr<SessionBundle> bundle(new SessionBundle);
  TF_QCHECK_OK(bundle_factory->CreateSessionBundle(bundle_path, &bundle));
  ...

  RunServer(FLAGS_port, std::move(bundle));

  return 0;
}
~~~

It uses the `SessionBundle` component of TensorFlow Serving.
`SessionBundleFactory::CreateSessionBundle()` loads an exported TensorFlow model
at the given path and creates a `SessionBundle` object for running inference
with the model. Typically, a default `tensorflow::SessionOptions` proto is given
when loading the model; a custom one can be passed via `SessionBundleConfig` if
desired.

With a small amount of extra code, you can arrange for the server to batch
groups of inference requests together into larger tensors, which tends to
improve throughput, especially on GPUs. To enable batching you simply populate
the `BatchingParameters` sub-message of the `SessionBundleConfig`, like so:

~~~c++
int main(int argc, char** argv) {
  ...
  BatchingParameters* batching_parameters =
      session_bundle_config.mutable_batching_parameters();
  batching_parameters->mutable_thread_pool_name()->set_value(
      "mnist_service_batch_threads");
  ...
}
~~~

This example sticks with default tuning parameters for batching; if you want to
adjust the maximum batch size, timeout threshold or the number of background
threads used for batched inference, you can do so by setting more values in
`BatchingParameters`. Note that the (simplified) batching API offered by
`SessionBundleFactory` requires a client thread to block while awaiting other
peer threads with which to form a batch -- gRPC promises to adjust the number of
client threads to keep things flowing smoothly. Lastly, the batcher's timeout
threshold bounds the amount of time a given request spends in the blocked state,
so a low request volume does not compromise latency.

Whether or not we enable batching, we wind up with a `SessionBundle`; let's look
at its definition in
[session_bundle.h](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/session_bundle/session_bundle.h):

~~~c++
struct SessionBundle {
  std::unique_ptr<tensorflow::Session> session;
  tensorflow::MetaGraphDef meta_graph_def;
};
~~~

`session` is, guess what, a TensorFlow session that has the original graph
with the needed variables properly restored. In other words, the trained model
is now held in `session` and is ready for running inference!

All you need to do now is bind inference input and output to the proper tensors
in the graph and invoke `session->run()`. But how do you know which tensors to
bind to? As you may have probably guessed, the answer is in the
`meta_graph_def`.

`tensorflow::MetaGraphDef` is the protobuf de-serialized from the `export.meta`
file above (see [meta_graph.proto](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/protobuf/meta_graph.proto)).
We add all needed description and metadata of a TensorFlow model export to its
extensible `collection_def`. In particular, it contains `Signatures` (see
[manifest.proto](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/session_bundle/manifest.proto))
that specifies the tensor to use.

~~~proto
// Signatures of model export.
message Signatures {
  // Default signature of the graph.
  Signature default_signature = 1;

  // Named signatures of the graph.
  map<string, Signature> named_signatures = 2;
};
~~~

Remember how we specified the model signature in `export` above? The
information eventually gets encoded here:

~~~proto
message ClassificationSignature {
  TensorBinding input = 1;
  TensorBinding classes = 2;
  TensorBinding scores = 3;
};
~~~

`TensorBinding` contains the tensor name that can be used for `session->run()`.
With that, we can run inference given a `SessionBundle`!

## Bring Up Inference Service

As you can see in `mnist_inference.cc`, `RunServer` in `main` brings up a gRPC
server that exports a single `Classify()` API. The implementation of the method
is straightforward, as each inference request is processed in the following
steps:

  1. Verify the input -- the server expects exactly one MNIST-format image for
  each inference request.

  2. Transform protobuf input to inference input tensor and create output
  tensor placeholder.

  3. Run inference. Note that in the `MnistServiceImpl` constructor you use
  `GetClassificationSignature()` to extract the signature of the model from
  'meta_graph_def` and verify that it is a classification signature as expected.
  With the extracted signature, the server can bind the input and output tensors
  properly and run the session.

  ~~~c++
    const tensorflow::Status status =
        bundle_->session->Run({{signature_.input().tensor_name(), input}},
                              {signature_.scores().tensor_name()}, {},
                              &outputs);
  ~~~

  4. Transform the inference output tensor to protobuf output.

To run it:

~~~shell
$>bazel build //tensorflow_serving/example:mnist_inference
$>bazel-bin/tensorflow_serving/example/mnist_inference --port=9000 /tmp/mnist_model/00000001
~~~

## Test The Server

We can use the provided [mnist_client](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/example/mnist_client.py) utility
to test the server. The client downloads MNIST test data, sends them as
requests to the server, and calculates the inference error rate.

To run it:

~~~shell
$>bazel build //tensorflow_serving/example:mnist_client
$>bazel-bin/tensorflow_serving/example/mnist_client --num_tests=1000 --server=localhost:9000
...
Inference error rate: 10.5%
~~~

We expect around 91% accuracy for the trained Softmax model and we get
10.5% inference error rate for the first 1000 test images. This confirms that
the server loads and runs the trained model successfully!
