# Serving TensorFlow models with custom ops

TensorFlow comes pre-built with an extensive library of ops and op kernels
(implementations) fine-tuned for different hardware types (CPU, GPU, etc.).
These operations are automatically linked into the TensorFlow Serving
ModelServer binary with no additional work required by the user. However, there
are two use cases that require the user to link in ops into the ModelServer
explicitly:

*   You have written your own custom op (ex. using
    [this guide](https://github.com/tensorflow/custom-op))
*   You are using an already implemented op that is not shipped with TensorFlow

Note: Starting in version 2.0, TensorFlow no longer distributes the contrib
module; if you are serving a TensorFlow program using contrib ops, use this
guide to link these ops into ModelServer explicitly.

Regardless of whether you implemented the op or not, in order to serve a model
with custom ops, you need access to the source of the op. This guide walks you
through the steps of using the source to make custom ops available for serving.
For guidance on implementation of custom ops, please refer to the
[tensorflow/custom-op](https://github.com/tensorflow/custom-op) repo.

Prerequisite: With Docker installed, you have cloned the TensorFlow Serving
[repository](https://github.com/tensorflow/serving) and your current working
directory is the root of the repo.

## Copy over op source into Serving project

In order to build TensorFlow Serving with your custom ops, you will first need
to copy over the op source into your serving project. For this example, you will
use
[tensorflow_zero_out](https://github.com/tensorflow/custom-op/tree/master/tensorflow_zero_out)
from the custom-op repository mentioned above.

Wihin the serving repo, create a `custom_ops` directory, which will house all
your custom ops. For this example, you will only have the
[tensorflow_zero_out](https://github.com/tensorflow/custom-op/tree/master/tensorflow_zero_out)
code.

```bash
mkdir tensorflow_serving/custom_ops
cp -r <custom_ops_repo_root>/tensorflow_zero_out tensorflow_serving/custom_ops
```

## Build static library for the op

In tensorflow_zero_out's
[BUILD file](https://github.com/tensorflow/custom-op/blob/master/tensorflow_zero_out/BUILD),
you see a target producing a shared object file (`.so`), which you would load
into python in order to create and train your model. TensorFlow Serving,
however, statically links ops at build time, and requires a `.a` file. So you
will add a build rule that produces this file to
`tensorflow_serving/custom_ops/tensorflow_zero_out/BUILD`:

```python
cc_library(
    name = 'zero_out_ops',
    srcs = [
        "cc/kernels/zero_out_kernels.cc",
        "cc/ops/zero_out_ops.cc",
    ],
    alwayslink = 1,
    deps = [
        "@org_tensorflow//tensorflow/core:framework",
    ]
)
```

## Build ModelServer with the op linked in

To serve a model that uses a custom op, you have to build the ModelServer binary
with that op linked in. Specifically, you add the `zero_out_ops` build target
created above to the ModelServer's `BUILD` file.

Edit `tensorflow_serving/model_servers/BUILD` to add your custom op build target
to `SUPPORTED_TENSORFLOW_OPS` which is inluded in the `server_lib` target:

```python
SUPPORTED_TENSORFLOW_OPS = [
    ...
    "//tensorflow_serving/custom_ops/tensorflow_zero_out:zero_out_ops"
]
```

Then use the Docker environment to build the ModelServer:

```bash
tools/run_in_docker.sh bazel build tensorflow_serving/model_servers:tensorflow_model_server
```

## Serve a model containing your custom op

You can now run the ModelServer binary and start serving a model that contains
this custom op:

```bash
tools/run_in_docker.sh -o "-p 8501:8501" \
bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server \
--rest_api_port=8501 --model_name=<model_name> --model_base_path=<model_base_path>
```

## Send an inference request to test op manually

You can now send an inference request to the model server to test your custom
op:

```bash
curl http://localhost:8501/v1/models/<model_name>:predict -X POST \
-d '{"inputs": [[1,2], [3,4]]}'
```

[This page](https://www.tensorflow.org/tfx/serving/api_rest#top_of_page)
contains a more complete API for sending REST requests to the model server.
