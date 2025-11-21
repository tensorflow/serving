# SignatureDefs in SavedModel for TensorFlow Serving

## Objective

This document provides examples for the intended usage of SignatureDefs in SavedModel
that map to TensorFlow Serving's APIs.

## Overview

A
[SignatureDef](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/meta_graph.proto)
defines the signature of a computation supported in a TensorFlow graph.
SignatureDefs aim to provide generic support to identify inputs and outputs of a
function and can be specified when building a
[SavedModel](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/builder.py).

## Background

[TF-Exporter](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/session_bundle/README.md)
and
[SessionBundle](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/session_bundle/session_bundle.h)
used
[Signatures](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/session_bundle/manifest.proto)
which are similar in concept but required users to distinguish between named and
default signatures in order for them to be retrieved correctly upon a load. For
those who previously used TF-Exporter/SessionBundle, `Signatures` in TF-Exporter
will be replaced by `SignatureDefs` in SavedModel.

## SignatureDef Structure

A SignatureDef requires specification of:

*   `inputs` as a map of string to TensorInfo.
*   `outputs` as a map of string to TensorInfo.
*   `method_name` (which corresponds to a supported method name in the loading
    tool/system).

Note that
[TensorInfo](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/meta_graph.proto#L194)
itself requires specification of name, dtype and tensor shape. While tensor
information is already present in the graph, it is useful to explicitly have the
TensorInfo defined as part of the SignatureDef since tools can then perform
signature validation, etc. without having to read the graph definition.

## Related constants and utils

For ease of reuse and sharing across tools and systems, commonly used constants
related to SignatureDefs that will be supported in TensorFlow Serving are defined as
constants. Specifically:

*   [Signature constants in
    Python](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/signature_constants.py).
*   [Signature constants in
    C++](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/cc/saved_model/signature_constants.h).

In addition, SavedModel provides a
[util](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/signature_def_utils.py)
to help build a signature-def.

## Sample structures

TensorFlow Serving provides high level APIs for performing inference. To enable these APIs,
models must include one or more SignatureDefs that define the exact TensorFlow
nodes to use for input and output. See below for examples of the specific
SignatureDefs that TensorFlow Serving supports for each API.

Note that TensorFlow Serving depends on the keys of each TensorInfo (in the inputs and
outputs of the SignatureDef), as well as the method_name of the SignatureDef.
The actual contents of the TensorInfo are specific to your graph.

### Classification SignatureDef

Classification SignatureDefs support structured calls to TensorFlow Serving's
Classification API. These prescribe that there must be an `inputs` Tensor, and
that there are two optional output Tensors: `classes` and `scores`, at least one
of which must be present.

```proto
signature_def: {
  key  : "my_classification_signature"
  value: {
    inputs: {
      key  : "inputs"
      value: {
        name: "tf_example:0"
        dtype: DT_STRING
        tensor_shape: ...
      }
    }
    outputs: {
      key  : "classes"
      value: {
        name: "index_to_string:0"
        dtype: DT_STRING
        tensor_shape: ...
      }
    }
    outputs: {
      key  : "scores"
      value: {
        name: "TopKV2:0"
        dtype: DT_FLOAT
        tensor_shape: ...
      }
    }
    method_name: "tensorflow/serving/classify"
  }
}
```

### Predict SignatureDef

Predict SignatureDefs support calls to TensorFlow Serving's Predict API. These
signatures allow you to flexibly support arbitrarily many input and output
Tensors. For the example below, the signature `my_prediction_signature` has a
single logical input Tensor `images` that are mapped to the actual Tensor in
your graph `x:0`.

Predict SignatureDefs enable portability across models. This means that you can
swap in different SavedModels, possibly with different underlying Tensor names
(e.g. instead of `x:0` perhaps you have a new alternate model with a Tensor
`z:0`), while your clients can stay online continuously querying the old and new
versions of this model without client-side changes.

Predict SignatureDefs also allow you to add optional additional Tensors to the
outputs, that you can explicitly query. Let's say that in addition to the output
key below of `scores`, you also wanted to fetch a pooling layer for debugging or
other purposes. In that case, you would simply add an additional Tensor with a
key like `pool` and appropriate value.

```proto
signature_def: {
  key  : "my_prediction_signature"
  value: {
    inputs: {
      key  : "images"
      value: {
        name: "x:0"
        dtype: ...
        tensor_shape: ...
      }
    }
    outputs: {
      key  : "scores"
      value: {
        name: "y:0"
        dtype: ...
        tensor_shape: ...
      }
    }
    method_name: "tensorflow/serving/predict"
  }
}
```

### Regression SignatureDef

Regression SignatureDefs support structured calls to TensorFlow Serving's
Regression API. These prescribe that there must be exactly one `inputs` Tensor,
and one `outputs` Tensor.

```proto
signature_def: {
  key  : "my_regression_signature"
  value: {
    inputs: {
      key  : "inputs"
      value: {
        name: "x_input_examples_tensor_0"
        dtype: ...
        tensor_shape: ...
      }
    }
    outputs: {
      key  : "outputs"
      value: {
        name: "y_outputs_0"
        dtype: DT_FLOAT
        tensor_shape: ...
      }
    }
    method_name: "tensorflow/serving/regress"
  }
}
```
