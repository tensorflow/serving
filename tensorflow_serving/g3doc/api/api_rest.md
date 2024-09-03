# RESTful API

In addition to
[gRPC APIs](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/prediction_service.proto)
TensorFlow ModelServer also supports RESTful APIs. This page describes these API
endpoints and an end-to-end [example](#example) on usage.

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

## Model status API

This API closely follows the
[`ModelService.GetModelStatus`](https://github.com/tensorflow/serving/blob/5369880e9143aa00d586ee536c12b04e945a977c/tensorflow_serving/apis/model_service.proto#L17)
gRPC API. It returns the status of a model in the ModelServer.

### URL

```
GET http://host:port/v1/models/${MODEL_NAME}[/versions/${VERSION}|/labels/${LABEL}]
```

Including `/versions/${VERSION}` or `/labels/${LABEL}` is optional. If omitted
status for all versions is returned in the response.

### Response format

If successful, returns a JSON representation of
[`GetModelStatusResponse`](https://github.com/tensorflow/serving/blob/5369880e9143aa00d586ee536c12b04e945a977c/tensorflow_serving/apis/get_model_status.proto#L64)
protobuf.

## Model Metadata API

This API closely follows the
[`PredictionService.GetModelMetadata`](https://github.com/tensorflow/serving/blob/5369880e9143aa00d586ee536c12b04e945a977c/tensorflow_serving/apis/prediction_service.proto#L29)
gRPC API. It returns the metadata of a model in the ModelServer.

### URL

```
GET http://host:port/v1/models/${MODEL_NAME}[/versions/${VERSION}|/labels/${LABEL}]/metadata
```

Including `/versions/${VERSION}` or `/labels/${LABEL}` is optional. If omitted
the model metadata for the latest version is returned in the response.

### Response format

If successful, returns a JSON representation of
[`GetModelMetadataResponse`](https://github.com/tensorflow/serving/blob/5369880e9143aa00d586ee536c12b04e945a977c/tensorflow_serving/apis/get_model_metadata.proto#L23)
protobuf.

## Classify and Regress API

This API closely follows the `Classify` and `Regress` methods of
[`PredictionService`](https://github.com/tensorflow/serving/blob/5369880e9143aa00d586ee536c12b04e945a977c/tensorflow_serving/apis/prediction_service.proto#L15)
gRPC API.

### URL

```
POST http://host:port/v1/models/${MODEL_NAME}[/versions/${VERSION}|/labels/${LABEL}]:(classify|regress)
```

Including `/versions/${VERSION}` or `/labels/${LABEL}` is optional. If omitted
the latest version is used.

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

`<value>` is a JSON number (whole or decimal), JSON string, or a JSON object
that represents binary data (see the [Encoding binary values](#encoding-binary-values)
section below for details). `<list>` is a list of such values. This
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

This API closely follows the
[`PredictionService.Predict`](https://github.com/tensorflow/serving/blob/5369880e9143aa00d586ee536c12b04e945a977c/tensorflow_serving/apis/prediction_service.proto#L23)
gRPC API.

### URL

```
POST http://host:port/v1/models/${MODEL_NAME}[/versions/${VERSION}|/labels/${LABEL}]:predict
```

Including `/versions/${VERSION}` or `/labels/${LABEL}` is optional. If omitted
the latest version is used.

### Request format

The request body for `predict` API must be JSON object formatted as follows:

```javascript
{
  // (Optional) Serving signature to use.
  // If unspecifed default serving signature is used.
  "signature_name": <string>,

  // Input Tensors in row ("instances") or columnar ("inputs") format.
  // A request can have either of them but NOT both.
  "instances": <value>|<(nested)list>|<list-of-objects>
  "inputs": <value>|<(nested)list>|<object>
}
```

#### Specifying input tensors in row format.

This format is similar to `PredictRequest` proto of gRPC API and the
[CMLE predict API](https://cloud.google.com/ml-engine/docs/v1/predict-request).
Use this format if all named input tensors have the **same 0-th dimension**. If
they don't, use the columnar format described later below.

In the row format, inputs are keyed to **instances** key in the JSON request.

When there is only one named input, specify the value of **instances** key to be
the value of the input:

```javascript
{
  // List of 3 scalar tensors.
  "instances": [ "foo", "bar", "baz" ]
}

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
     "tag": "foo",
     "signal": [1, 2, 3, 4, 5],
     "sensor": [[1, 2], [3, 4]]
   },
   {
     "tag": "bar",
     "signal": [3, 4, 1, 2, 5]],
     "sensor": [[4, 5], [6, 8]]
   }
 ]
}
```

Note, each named input ("tag", "signal", "sensor") is implicitly assumed have
same 0-th dimension (*two* in above example, as there are *two* objects in the
*instances* list). If you have named inputs that have different 0-th dimension,
use the columnar format described below.

#### Specifying input tensors in column format.

Use this format to specify your input tensors, if individual named inputs do not
have the same 0-th dimension or you want a more compact representation. This
format is similar to the `inputs` field of the gRPC
[`Predict`](https://github.com/tensorflow/serving/blob/a52e8181144a5d6acc96b3d57328c7f49f113ea9/tensorflow_serving/apis/predict.proto#L21)
request.

In the columnar format, inputs are keyed to **inputs** key in the JSON request.

The value for **inputs** key can either a single input tensor or a map of input
name to tensors (listed in their natural nested form). Each input can have
arbitrary shape and need not share the/ same 0-th dimension (aka batch size) as
required by the row format described above.

Columnar representation of the previous example is as follows:

```javascript
{
 "inputs": {
   "tag": ["foo", "bar"],
   "signal": [[1, 2, 3, 4, 5], [3, 4, 1, 2, 5]],
   "sensor": [[[1, 2], [3, 4]], [[4, 5], [6, 8]]]
 }
}
```

Note, **inputs** is a JSON object and not a list like **instances** (used in the
row representation). Also, all the named inputs are specified together, as
opposed to unrolling them into individual rows done in the row format described
previously. This makes the representation compact (but maybe less readable).

### Response format

The `predict` request returns a JSON object in response body.

A request in [row format](#specifying-input-tensors-in-row-format) has response
formatted as follows:

```javascript
{
  "predictions": <value>|<(nested)list>|<list-of-objects>
}
```

If the output of the model contains only one named tensor, we omit the name and
`predictions` key maps to a list of scalar or list values. If the model outputs
multiple named tensors, we output a list of objects instead, similar to the
request in row-format mentioned above.

A request in [columnar format](#specifying-input-tensors-in-column-format) has
response formatted as follows:

```javascript
{
  "outputs": <value>|<(nested)list>|<object>
}
```

If the output of the model contains only one named tensor, we omit the name and
`outputs` key maps to a list of scalar or list values. If the model outputs
multiple named tensors, we output an object instead. Each key of this object
corresponds to a named output tensor. The format is similar to the request in
column format mentioned above.

#### Output of binary values

TensorFlow does not distinguish between non-binary and binary strings. All are
`DT_STRING` type. Named tensors that have **`_bytes`** as a suffix in their name
are considered to have binary values. Such values are encoded differently as
described in the [encoding binary values](#encoding-binary-values) section
below.

## JSON mapping

The RESTful APIs support a canonical encoding in JSON, making it easier to share
data between systems. For supported types, the encodings are described on a
type-by-type basis in the table below. Types not listed below are implied to be
unsupported.

[TF Data Type](https://www.tensorflow.org/versions/r1.1/programmers_guide/dims_types#data_types) | [JSON Value](http://json.org/) | JSON example                       | Notes
------------------------------------------------------------------------------------------------ | ------------------------------ | ---------------------------------- | -----
DT_BOOL                                                                                          | true, false                    | *true, false*                      |
DT_STRING                                                                                        | string                         | *"Hello World!"*                   | If `DT_STRING` represents binary bytes (e.g. serialized image bytes or protobuf), encode these in Base64. See [Encoding binary values](#encoding-binary-values) for more info.
DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64                            | number                         | *1, -10, 0*                        | JSON value will be a decimal number.
DT_FLOAT, DT_DOUBLE                                                                              | number                         | *1.1, -10.0, 0, `NaN`, `Infinity`* | JSON value will be a number or one of the special token values - `NaN`, `Infinity`, and `-Infinity`. See [JSON conformance](#json-conformance) for more info. Exponent notation is also accepted.

## Floating Point Precision

JSON has a single number data type. Thus it is possible to provide a value
for an input that results in a loss of precision. For instance, if the
input `x` is a `float` data type, and the input `{"x": 1435774380}` is
sent to the model running on hardware based on the IEEE 754 floating point
standard (e.g. Intel or AMD), then the value will be silently converted by the
underyling hardware to `1435774336` since `1435774380` cannot be exactly
represented in a 32-bit floating point number. Typically, the inputs
to serving should be the same distribution as training, so this generally
won't be problematic because the same conversions happened at training time.
However, in case full precision is needed, be sure to use an underlying data
type in your model that can handle the desired precision and/or consider
client-side checking.

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
  ]
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
(due to `NaN` and `Infinity` tokens mixed with actual numbers). To correctly
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

Download the `half_plus_three` model from
[git repository](https://github.com/tensorflow/serving):

```shell
$ mkdir -p /tmp/tfserving
$ cd /tmp/tfserving
$ git clone --depth=1 https://github.com/tensorflow/serving
```

We will use Docker to run the ModelServer. If you want to install ModelServer
natively on your system, follow
[setup instructions](https://www.tensorflow.org/tfx/serving/setup) to install
instead, and start the ModelServer with `--rest_api_port` option to export
REST API endpoint (this is not needed when using Docker).

```shell
$ cd /tmp/tfserving
$ docker pull tensorflow/serving:latest
$ docker run --rm -p 8501:8501 \
    --mount type=bind,source=$(pwd),target=$(pwd) \
    -e MODEL_BASE_PATH=$(pwd)/serving/tensorflow_serving/servables/tensorflow/testdata \
    -e MODEL_NAME=saved_model_half_plus_three -t tensorflow/serving:latest
...
.... Exporting HTTP/REST API at:localhost:8501 ...
```

### Make REST API calls to ModelServer

In a different terminal, use the `curl` tool to make REST API calls.

Get status of the model as follows:

```
$ curl http://localhost:8501/v1/models/saved_model_half_plus_three
{
 "model_version_status": [
  {
   "version": "123",
   "state": "AVAILABLE",
   "status": {
    "error_code": "OK",
    "error_message": ""
   }
  }
 ]
}
```

A `predict` call would look as follows:

```shell
$ curl -d '{"instances": [1.0,2.0,5.0]}' -X POST http://localhost:8501/v1/models/saved_model_half_plus_three:predict
{
    "predictions": [3.5, 4.0, 5.5]
}
```

And a `regress` call looks as follows:

```shell
$ curl -d '{"signature_name": "tensorflow/serving/regress", "examples": [{"x": 1.0}, {"x": 2.0}]}' \
  -X POST http://localhost:8501/v1/models/saved_model_half_plus_three:regress
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
