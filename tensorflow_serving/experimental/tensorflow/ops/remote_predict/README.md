# Remote Predict TensorFlow Operator

_Warning: This is an experimental feature, not yet supported by TensorFlow
Serving team and may change or be removed at any point. While we would love to
hear about your interest and discuss your use cases for this op via
[Github Issues](https://github.com/tensorflow/serving/issues), we cannot offer
debugging help and support for this feature at the moment as many pieces needed
to productionize its usage are not currently available. We look forward to
hearing your preliminary thoughts and feedback as we continue to assess the need
for this feature and iterate on its design._

## Motivation and Use Case

The Remote Predict Op (RPOp) is an experimental TensorFlow operator that enables
users to make a
[Predict RPC](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/prediction_service.proto#L23)
from within a TensorFlow graph executing on machine A to another graph hosted by
TensorFlow Serving on machine B. This capability unlocks the following two
use-cases:

1) *Serving large models*: When model size grows beyond what can be loaded into
the host machine's RAM, we can split the model into subgraphs, hosted on
multiple machines, and use this op to distribute the inference over the
subgraphs on the various machines.

2) *Re-using model sub-graphs*: In cases where we have N models that share the
same base (as in Transfer Learning use cases) or otherwise have significant
common subgraphs or shared embeddings, instead of embedding the shared component
in each graph, we can host them centrally, and have the different consumer
graphs use the RPOp to remote call into them.

For this guide, we will do a deeper walk-through of the first use case.

### Serving Large Models

Many machine learning models have to make inferences over items in large
corpuses of potentially billions of items. The elements of these corpuses are
often represented using large embedding matrices, which in the most extreme
cases are too large to fit in the RAM of the host machine.

Using RPOp, we can shard the embedding matrix and split it over multiple
machines, all running TensorFlow Serving to serve their embedding shard. We
could then have a master graph that processes the input request, farms out
requests to the various leaf shards, post-processes their responses and returns
the final response to the user.

## Configuration

This op uses a combination of attributes (set at model export time) and inputs
(set at inference time) to configure remote inference.

Some notable configurable attributes include:

*   `target_address`: Address of the server hosting the remote graph.
*   `model_name`: The name of the remote TF graph.
*   `model_version`: the target model version for the Predict call. When unset,
    the default value (-1) implies the latest available version should be used.
*   `max_rpc_deadline_millis`: The rpc deadline for remote predict. Of course if
    the incoming RPC times out before this deadline is reached, the client
    should timeout the incoming RPC to this server.
*   `output_types`: A list, equal in length to output_tensors, of types of the
    output tensors.

The inputs that the op expects at inference time are:

*   `input_tensor_aliases`: Tensor of strings for the input tensor alias names
    to supply to the RemotePredict call.
*   `input_tensors`: List of tensors to provide as input, which should be equal
    in length to 'input_tensor_aliases'.
*   `output_tensor_aliases`: Tensor of strings for the output tensor alias names
    to supply to the Predict call.

And the outputs that the op will provide are:

*   `status_code`: Returns the status code of the RPC, converting
    tensorflow::error::Code to its int value; 0 means OK.
*   `status_error_message`: Returns the error message contained in the RPC
    status.
*   `output_tensors`: Tensors returned by the Predict call on the remote graph.

It's worth noting that the RPOp kernel implementation is templated on a
PredictionServiceStubType, which allows users to easily extend it to support RPC
frameworks other than gRPC, for which this op comes with out of the box support.

## Usage

To demonstrate usage of this feature, we'll serve a modified version of the
[half plus two model](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_two.py)
where instead of directly computing `y = a * x + b; a = 0.5, b = 2`, we rely on
a remote graph (a vanilla half plus two model) to supply us the value for `b` in
the first tensor in its output. The code for this example model can be found
[here](../../../example/half_plus_two_with_rpop.py).

To ease the set up and not get into networking-environment-specific concerns,
we'll demonstrate deploying both of these on the same TF Serving instance,
utilizing the host's loopback interface.

### Export Models and Model Config

Note: The following assumes your current working directory is the root of the
repository.

First, use [this script](../../../example/half_plus_two_with_rpop.py) to export
the model containing the RPOp. By default, it will be exported to
`/tmp/half_plus_two_with_rpop`. Read through the script to see how to configure
the parameters of the RPOp:

`tools/run_in_docker.sh bazel run
tensorflow_serving/experimental/example:half_plus_two_with_rpop --
--target_address=localhost:8500`

Then, copy over the vanilla half plus two
[model](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_two_cpu).

`cp -r
tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_two_cpu/
/tmp/half_plus_two`

Finally, create a model config file `/tmp/config_file.txt` containing the
following:

```
model_config_list {
  config: {
    name: "half_plus_two"
    base_path: "/tmp/half_plus_two"
    model_platform: "tensorflow"
  }
  config: {
    name: "half_plus_two_with_rpop"
    base_path: "/tmp/half_plus_two_with_rpop"
    model_platform: "tensorflow"
  }
}
```

### Build and Run TF Serving

You will then need to build TF Serving with the RPOp op and kernel linked in.

Add the following two lines to the list of deps of the `server_lib` C++ library
build rule in model server's
[BUILD file](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/model_servers/BUILD)
to link the op and kernel into the model server binary.

```
cc_library(
    name = "server_lib",
    ...
    deps = [ ...
    "//tensorflow_serving/experimental/tensorflow/ops/remote_predict:remote_predict_op_kernel",
    "//tensorflow_serving/experimental/tensorflow/ops/remote_predict:remote_predict_ops_cc",
    ] + SUPPORTED_TENSORFLOW_OPS,
)
```

Then use this command to build the TF Serving binary, with the op linked in,
using Docker:

`tools/run_in_docker.sh bazel build
tensorflow_serving/model_servers:tensorflow_model_server`

### Deploy Models

Now, you can start TF Serving:

`bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=8500
--rest_api_port=8501 --model_config_file=/tmp/config_file.txt`

### Make Inference Calls

You may now send inference requests to this model:

`curl -d '{"instances": [1.0, 2.0, 5.0]}' -X POST
http://localhost:8501/v1/models/half_plus_two_with_rpop:predict`

## Caveat & Feedback

As mentioned in the warning, this op is only a part of a larger set of
infrastructure needed to enable distributed serving in production settings,
including support for determining reasonable graph partition points, executing
the graph splits, injection of the RPOp(s), coordinating new model version
rollouts, load balancing, autoscaling and more. We look forward to hearing your
feedback and thoughts on whether this op is a good fit for your production use
cases to help us better understand the value in releasing the additional tooling
and infrastructure needed to productionize such a set up.
