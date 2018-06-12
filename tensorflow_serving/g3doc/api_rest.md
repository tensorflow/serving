# RESTful API

In addition to [gRPC
APIs](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/prediction_service.proto)
TensorFlow ModelServer also supports RESTful APIs for classification, regression
and prediction on TensorFlow models. This page describes these API endpoints and
format of request/response involved in using them.

TensorFlow ModelServer running on `host:port` accepts following REST API
requests:

```
POST http://host:port/<URI>:<VERB>

URI: /v1/models/${MODEL_NAME}[/versions/${MODEL_VERSION}]
VERB: classify|regress|predict
```

`/versions/${MODEL_VERSION}` is optional. If omitted the latest version is used.

This API closely follows the gRPC version of
[`PredictionService`](https://github.com/tensorflow/serving/blob/5369880e9143aa00d586ee536c12b04e945a977c/tensorflow_serving/apis/prediction_service.proto#L15)
API.

Examples of request URLs:

```
http://host:port/v1/models/iris:classify
http://host:port/v1/models/mnist/versions/314:predict
```

The request and response is a JSON object. The composition of this object
depends on the request type or verb. See the API specific sections below for
details.

In case of error, all APIs will return a JSON object in the response body with
`error` as key and the error message as the value:

```javascript
{
  "error": <error message string>
}
```

## Classify and Regress API

### Request format

The request body for the `classify` and `regress` APIs must be a JSON object
formatted as follows:

```javascript
{
  // Optional: serving signature to use.
  // If unspecifed default serving signature is used.
  "signature_name": <string>,

  // Optional: Common context shared by all examples.
  // Features that appear here MUST NOT appear in examples (below).
  "context": {
    "<feature_name3>": <value>|<list>
    "<feature_name4>": <value>|<list>
  },

  // List of Example objects
  "examples": [
    {
      // Example 1
      "<feature_name1>": <value>|<list>,
      "<feature_name2>": <value>|<list>,
      ...
    },
    {
      // Example 2
      "<feature_name1>": <value>|<list>,
      "<feature_name2>": <value>|<list>,
      ...
    }
    ...
  ]
}
```

`<value>` is a JSON number (whole or decimal) or string, and `<list>` is a list
of such values. See [Encoding binary values](#encoding-binary-values) section
below for details on how to represent a binary (stream of bytes) value. This
format is similar to gRPC's `ClassificationRequest` and `RegressionRequest`
protos. Both versions accept list of
[`Example`](https://github.com/tensorflow/tensorflow/blob/92e6c3e4f5c1cabfda1e61547a6a1b268ef95fa5/tensorflow/core/example/example.proto#L13)
objects.

### Response format

A `classify` request returns a JSON object in the response body, formatted as
follows:

```javascript
{
  "result": [
    // List of class label/score pairs for first Example (in request)
    [ [<label1>, <score1>], [<label2>, <score2>], ... ],

    // List of class label/score pairs for next Example (in request)
    [ [<label1>, <score1>], [<label2>, <score2>], ... ],
    ...
  ]
}
```

`<label>` is a string (which can be an empty string `""` if the model does not
have a label associated with the score). `<score>` is a decimal (floating point)
number.

The `regress` request returns a JSON object in the response body, formatted as
follows:

```javascript
{
  // One regression value for each example in the request in the same order.
  "result": [ <value1>, <value2>, <value3>, ...]
}
```

`<value>` is a decimal number.

Users of gRPC API will notice the similarity of this format with
`ClassificationResponse` and `RegressionResponse` protos.

## Predict API

### Request format

The request body for `predict` API must be JSON object formatted as follows:

```javascript
{
  // Optional: serving signature to use.
  // If unspecifed default serving signature is used.
  "signature_name": <string>,

  // List of tensors (each element must be of same shape and type)
  "instances": [ <value>|<(nested)list>|<object>, ... ]
}
```

This format is similar to `PredictRequest` proto of gRPC API and the [CMLE
predict API](https://cloud.google.com/ml-engine/docs/v1/predict-request).

When there is only one named input, the list items are expected to be scalars
(number/string):

```javascript
{
  "instances": [ "foo", "bar", "baz" ]
}
```

or lists of these primitive types.

```javascript
{
  // List of 2 tensors each of [1, 2] shape
  "instances": [ [[1, 2]], [[3, 4]] ]
}
```

Tensors are expressed naturally in nested notation since there is no need to
manually flatten the list.

For multiple named inputs, each item is expected to be an object containing
input name/tensor value pair, one for each named input. As an example, the
following is a request with two instances, each with a set of three named input
tensors:

```javascript
{
 "instances": [
   {
     "tag": ["foo"]
     "signal": [1, 2, 3, 4, 5]
     "sensor": [[1, 2], [3, 4]]
   },
   {
     "tag": ["bar"]
     "signal": [3, 4, 1, 2, 5]]
     "sensor": [[4, 5], [6, 8]]
   },
 ]
}
```

See the [Encoding binary values](#encoding-binary-values) section below for
details on how to represent a binary (stream of bytes) value.

### Response format

The `predict` request returns a JSON object in response body, formatted as
follows:

```javascript
{
  "predictions": [ <value>|<(nested)list>|<object>, ...]
}
```

If the output of the model contains only one named tensor, we omit the name and
`predictions` key maps to a list of scalar or list values. If the model outputs
multiple named tensors, we output a list of objects instead, similar to the
request format mentioned above.

Named tensors that have `_bytes` as a suffix in their name are considered to
have binary values. Such values are encoded differently as described in the
[encoding binary values](#encoding-binary-values) section below.

## JSON mapping

The RESTful APIs support a canonical encoding in JSON, making it easier to share
data between systems. For supported types, the encodings are described on a
type-by-type basis in the table below. Types not listed below are implied to be
unsupported.

[TF Data Type](https://www.tensorflow.org/versions/r1.1/programmers_guide/dims_types#data_types) | [JSON Value](http://json.org/) | JSON example                       | Notes
------------------------------------------------------------------------------------------------ | ------------------------------ | ---------------------------------- | -----
DT_BOOL                                                                                          | true, false                    | *true, false*                      |
DT_STRING                                                                                        | string                         | *"Hello World!"*                   |
DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64                            | number                         | *1, -10, 0*                        | JSON value will be a decimal number.
DT_FLOAT, DT_DOUBLE                                                                              | number                         | *1.1, -10.0, 0, `NaN`, `Infinity`* | JSON value will be a number or one of the special token values - `NaN`, `Infinity`, and `-Infinity`. See [JSON conformance](#json-conformance) for more info. Exponent notation is also accepted.

## Encoding binary values

JSON uses UTF-8 encoding. If you have input feature or tensor values that need
to be binary (like image bytes), you *must* Base64 encode the data and
encapsulate it in a JSON object having `b64` as the key as follows:

```javascript
{ "b64": <base64 encoded string> }
```

You can specify this object as a value for an input feature or tensor. The same
format is used to encode output response as well.

A classification request with `image` (binary data) and `caption` features is
shown below:

```javascript
{
  "signature_name": "classify_objects",
  "examples": [
    {
      "image": { "b64": "aW1hZ2UgYnl0ZXM=" },
      "caption": "seaside"
    },
    {
      "image": { "b64": "YXdlc29tZSBpbWFnZSBieXRlcw==" },
      "caption": "mountains"
    }
  }
}
```

## JSON conformance

Many feature or tensor values are floating point numbers. Apart from finite
values (e.g. 3.14, 1.0 etc.) these can have `NaN` and non-finite (`Infinity` and
`-Infinity`) values. Unfortunately the JSON specification ([RFC
7159](https://tools.ietf.org/html/rfc7159)) does **NOT** recognize these values
(though the JavaScript specification does).

The REST API described on this page allows request/response JSON objects to have
such values. This implies that requests like the following one are valid:

```javascript
{
  "example": [
    {
      "sensor_readings": [ 1.0, -3.14, Nan, Infinity ]
    }
  ]
}
```

A (strict) standards compliant JSON parser will reject this with a parse error
(due to `NaN` and `Inifinity` tokens mixed with actual numbers). To correctly
handle requests/responses in your code, use a JSON parser that supports these
tokens.

`NaN`, `Infinity`, `-Infinity` tokens are recognized by
[proto3](https://developers.google.com/protocol-buffers/docs/proto3#json),
Python [JSON](https://docs.python.org/3/library/json.html) module and JavaScript
language.

## Example

We can use the toy
[half_plus_three](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_three/00000123)
model to see REST APIs in action.

### Start ModelServer with the REST API endpoint

Follow [setup instructions](https://www.tensorflow.org/serving/setup) to install
TensorFlow ModelServer on your system. Then download the `half_plus_three` model
from [git repository](https://github.com/tensorflow/serving):

```shell
$ mkdir -p /tmp/tfserving
$ cd /tmp/tfserving
$ git clone --depth=1 https://github.com/tensorflow/serving
```

Start the ModelServer with `--rest_api_port` option to export REST API endpoint:

```shell
$ tensorflow_model_server --rest_api_port=8501 \
   --model_name=half_plus_three \
   --model_base_path=$(pwd)/serving/tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_three/
```

### Make REST API calls to ModelServer

In a different terminal, use the `curl` tool to make REST API calls. A `predict`
call would look as follows:

```shell
$ curl -d '{"instances": [1.0,2.0,5.0]}' -X POST http://localhost:8501/v1/models/half_plus_three:predict
{
    "predictions": [3.5, 4.0, 5.5]
}
```

And a `regress` call looks as follows:

```shell
$ curl -d '{"signature_name": "tensorflow/serving/regress", "examples": [{"x": 1.0}, {"x": 2.0}]}' \
  -X POST http://localhost:8501/v1/models/half_plus_three:regress
{
    "results": [3.5, 4.0]
}
```

Note, `regress` is available on a non-default signature name and must be
specified explicitly. An incorrect request URL or body returns an HTTP error
status.

```shell
$ curl -i -d '{"instances": [1.0,5.0]}' -X POST http://localhost:8501/v1/models/half:predict
HTTP/1.1 404 Not Found
Content-Type: application/json
Date: Wed, 06 Jun 2018 23:20:12 GMT
Content-Length: 65

{ "error": "Servable not found for request: Latest(half)" }
$
```
