# Serving a TensorFlow Model

This tutorial shows you how to use TensorFlow Serving components to export a
trained TensorFlow model and use the standard tensorflow_model_server to serve
it. If you are already familiar with TensorFlow Serving, and you want to know
more about how the server internals work, see the
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
[main.cc](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/model_servers/main.cc)
which is the standard TensorFlow model server that discovers new exported
models and runs a [gRPC](http://www.grpc.io) service for serving them.

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
from tensorflow.contrib.session_bundle import exporter
...
export_path = sys.argv[-1]
print 'Exporting trained model to', export_path
saver = tf.train.Saver(sharded=True)
model_exporter = exporter.Exporter(saver)
model_exporter.init(
    sess.graph.as_graph_def(),
    named_graph_signatures={
        'inputs': exporter.generic_signature({'images': x}),
        'outputs': exporter.generic_signature({'scores': y})})
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

  * `named_graph_signatures=...` specifies a model export **signature**.
  Signature specifies what type of model is being exported, and the input/output
  tensors to bind to when running inference. In this case, you use
  `inputs` and `outputs` as keys for `exporter.generic_signature` as such a
  signature is supported by the standard `tensorflow_model_server`:

    * `{'images': x}` specifies the input tensor binding.

    * `{'scores': y}` specifies the scores tensor binding.

    * `images` and `scores` are tensor alias names. They can be whatever
    unique strings you want, and they will become the logical names of tensor
    `x` and `y` that you refer to for tensor binding when sending prediction
    requests later. For instance, if `x` refers to the tensor with name
    'long_tensor_name_foo' and `y` refers to the tensor with name
    'generated_tensor_name_bar', `exporter.generic_signature` will store
    tensor logical name to real name mapping ('images' -> 'long_tensor_name_foo'
    and 'scores' -> 'generated_tensor_name_bar') and allow user to refer to
    these tensors with their logical names when running inference.

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

## Load Exported Model With Standard TensorFlow Model Server

~~~shell
$>bazel build //tensorflow_serving/model_servers:tensorflow_model_server
$>bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_name=mnist --model_base_path=/tmp/mnist_model/
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
