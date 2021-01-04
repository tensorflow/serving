/* Copyright 2018 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_SERVING_UTIL_JSON_TENSOR_H_
#define TENSORFLOW_SERVING_UTIL_JSON_TENSOR_H_

#include <functional>
#include <string>

#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow_serving/apis/classification.pb.h"
#include "tensorflow_serving/apis/predict.pb.h"
#include "tensorflow_serving/apis/regression.pb.h"

namespace tensorflow {
namespace serving {

// Format of input tensors in predict request.
// See comments for FillPredictRequestFromJson() below for more details.
enum class JsonPredictRequestFormat {
  kInvalid,
  kRow,
  kColumnar,
};

// Fills PredictRequest proto from a JSON object.
//
// `json` string is parsed to create TensorProtos based on the type map returned
// via `get_tensorinfo_map` and added to `PredictRequest.inputs`. Both maps are
// keyed by the name/alias of the tensor as it appears in the TensorFlow graph,
// and must contain at-least one entry. The name to <type> (e.g. DT_FLOAT etc.)
// mapping is typically part of the graph metadata.
//
// Following fields of the request proto are filled in:
//
//   `model_spec.signature_name` (string)
//   `inputs` (map string -> tensors)
//
// The JSON object is expected to be formatted as follows:
//
// {
//   "signature_name": <string>
//   ("instances"|"inputs"): [ <value>|<(nested)list>|<object>, ... ]
// }
//
// The "signature_name" is *optional* (if not specified, default serving
// signature is used). The "instances" or "inputs" represents list of tensors
// (read further on the formatting of these tensors below). Any other keys in
// the top-level JSON object are ignored.
//
// "instances" is used to format input tensors in "row" format and "inputs"
// is used to format them in "columnar" format. The former is easy to read
// but requires all inputs to have same 0-th dimension whereas the latter
// can represent input tensors with varying sizes.
//
// === Notes on formatting of "instances" (row format) in the JSON ===
//
// The "instances" represents a list of tensors (all of same shape+type), and
// this function builds a stack of these tensors (represented as tensor). If the
// list has tensors of rank-R, the output tensor is of rank R+1.
//
// The "instances" (key) maps to list of tensors (value). Each element of the
// list is of same type (and nesting, in case listoflists).
//
// {
//   "instances": [ <value>|<(nested)list>|<object>, ... ]
// }
//
// This formatting is similar to CMLE predict API:
// https://cloud.google.com/ml-engine/docs/v1/predict-request
//
// o When there is only one named input (tensorinfo_map has only one key),
//   the list items are expected to be scalars (number/string) or lists of
//   these primitive types.
//
//   {
//     "instances": [ "foo", "bar", "baz" ]
//   }
//
//   {
//     "instances": [ [[1, 2]], [[3], [4]] ]
//   }
//
// o For multiple named inputs (tensorinfo_map has >1 keys), each item is
//   expected to be an object containing key(name)/value(tensor) pairs, one
//   for each named input. Representing 2 instaces of a set of 3 named input
//   tensors would look as follows:
//
//   {
//     "instances": [
//       {
//         "tag": ["foo"],
//         "signal": [1, 2, 3, 4, 5],
//         "sensor": [[1, 2], [3, 4]]
//       },
//       {
//         "tag": ["bar"],
//         "signal": [3, 4, 1, 2, 5],
//         "sensor": [[4, 5], [6, 8]]
//       }
//     ]
//   }
//
// o Default encoding of strings is UTF-8. To express binary data (like image
//   bytes) use a JSON object '{ "b64": "<base64-encoded-data>" }' instead of
//   a raw string. Following example shows list of 2 binary strings:
//
//   {
//     "instances": [ { "b64" : "aGVsbG8=" }, { "b64": "d29ybGQ=" } ]
//   }
//
// === Notes on formatting of "inputs" (columnar format) in the JSON ===
//
// The value for "inputs" key is either a single input tensor or a map
// of input name to tensor values. Each input can have arbitrary shape
// and need not share the same 0-th dimension as required by the row
// format described above.
//
// This format is similar to the `inputs` field in request proto of
// gRPC Predict API. And compact compared to the row format
//
// A sample (compare this to "instances" above) is as follows:
// {
//   "inputs": {
//     "tag": ["foo", "bar"],
//     "signal": [1, 2, 3, 4, 5, 3, 4, 1, 2, 4],
//     "sensor": [[1, 2], [3, 4], [4, 5], [6, 8]]
//   }
// }
tensorflow::Status FillPredictRequestFromJson(
    const absl::string_view json,
    const std::function<tensorflow::Status(
        const string&, ::google::protobuf::Map<string, tensorflow::TensorInfo>*)>&
        get_tensorinfo_map,
    PredictRequest* request, JsonPredictRequestFormat* format);

// Fills ClassificationRequest proto from a JSON object.
//
// `json` string is parsed to create `Example` protos and added to
// `ClassificationRequest.inputs`.
//
// Following fields of the request proto are filled in:
//
//   `model_spec.signature_name` (string)
//   `input` (list of example protos)
//
// The JSON object is expected to be formatted as follows:
//
// {
//   "signature_name": <string>
//
//   "context": {
//     // Common (example) context shared by all examples.
//     "<feature_name3>": <value>|<list>
//     "<feature_name4>": <value>|<list>
//   },
//
//   "examples": [
//     {
//       // Example 1
//       "<feature_name>": <value>|<list>
//       "<feature_name2>": <value>|<list>
//     },
//     {
//       // Example 2
//       "<feature_name>": <value>|<list>
//       "<feature_name2>": <value>|<list>
//     },
//   ]
// }
//
// The "signature_name" is *optional* (if not specified, default serving
// signature is used). "context" is also optional. "exampless" represents
// list of examples. Any other keys in the top-level JSON object are ignored.
tensorflow::Status FillClassificationRequestFromJson(
    const absl::string_view json, ClassificationRequest* request);

// Same as FillClassificationRequestFromJson() but fills a RegressionRequest.
// See comments above on how Example protos are expected to be formatted.
tensorflow::Status FillRegressionRequestFromJson(const absl::string_view json,
                                                 RegressionRequest* request);

// Make JSON object from TensorProtos.
//
// `tensor_map` contains a map of name/alias tensor names (as it appears in the
// TensorFlow graph) to tensor protos. The output `json` string is JSON object
// containing all the tensors represented by these tensor protos. The
// composition of output JSON depends on `format` (read further for details).
//
// In case of error the contents of output `json` should not be used.
//
// Tensors containing binary data (e.g. image bytes) are base64 encoded in the
// JSON object. The name/alias for these tensors MUST have "_bytes" as suffix,
// to ensure JSON has correct (base64) encoding, otherwise the resulting JSON
// may not represent the output correctly and/or may not be parsable.
//
// Formatting when `format` is `kRow`:
//
//   Tensors appear as list (with "batch" size elements) keyed by "predictions"
//   in the JSON object:
//
//     {
//       "predictions": [ <value>|<(nested)list>|<object>, ...]
//     }
//
//   If `tensor_map` contains only one key/named tensor, the "predictions" key
//   contains a array of <value> or <(nested)list> otherwise it is an array of
//   JSON objects. See comments for FillPredictRequestFromJson() above for
//   formatting details (and unit-test of examples).
//
//   The first dimension in each of these tensors is assumed to be the "batch"
//   size and it is expected that all tensors have the *same* batch size --
//   otherwise we return an error.
//
//   Note, this formatting is similar to CMLE predict API:
//   https://cloud.google.com/ml-engine/docs/v1/predict-request#response-body
//
// Formatting when `format` is `kColumnar`
//
//   Tensors appear as list (when there is only one named output) or as a JSON
//   object, with one key-value pair for each named input, keyed by "outputs"
//   in the JSON object.
//
//   Unlike `kRow` format (see above) each named tensor can have different first
//   dimension. This format is compact and matches closely with the response
//   proto of the gRPC predict API.
//
//   Only one named output:
//     {
//       "outputs": [ <value>|<(nested)list> ]
//     }
//
//   Multiple named output:
//     {
//       "outputs": {
//         "named_output_foo": [ <value>|<(nested)list> ],
//         "named_output_bar": [ <value>|<(nested)list> ],
//         ...
//       }
//     }
//
//   See comments for FillPredictRequestFromJson() above for formatting details
//   (and unit-test of examples).
tensorflow::Status MakeJsonFromTensors(
    const ::google::protobuf::Map<string, tensorflow::TensorProto>& tensor_map,
    JsonPredictRequestFormat format, string* json);

// Make JSON object from ClassificationResult proto.
//
// The output JSON object is formatted as follows:
//
// {
//   "result": [
//      // List of class label/score pairs for first Example (in request)
//      [ [ <label1>, <score1> ], [ <label2>, <score2> ], ... ],
//
//      // List of class label/score pairs for next Example (in request)
//      [ [ <label1>, <score1> ], [ <label2>, <score2> ], ... ],
//      ...
//   ]
// }
//
// Note, all the results are keyed by "result" key in the JSON object, and
// they are ordered to match the input/request ordering of Examples. <label>
// is a string, and <score> is a float number. <label> can be missing from
// the results of graph evaluation, and these would appear as empty strings
// in above JSON.
tensorflow::Status MakeJsonFromClassificationResult(
    const ClassificationResult& result, string* json);

// Make JSON object from RegressionResult proto.
//
// The output JSON object is formatted as follows:
//
// {
//   // One regression value for each example in the request in the same order.
//   "result": [ <value1>, <value2>, <value3>, ...]
// }
//
// Note, all the results are keyed by "result" key in the JSON object.
// <value> is a float number.
tensorflow::Status MakeJsonFromRegressionResult(const RegressionResult& result,
                                                string* json);

// Make JSON object from Status.
//
// The output JSON object is formatted as follows:
//
// {
//   "error": "<status error message>"
// }
//
// If `status` is OK then we do not append anything to output `json`.
void MakeJsonFromStatus(const tensorflow::Status& status, string* json);

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_UTIL_JSON_TENSOR_H_
