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

#include "tensorflow_serving/util/json_tensor.h"

#include <cstdlib>
#include <limits>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"
#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow_serving/apis/input.pb.h"
#include "tensorflow_serving/apis/model.pb.h"

namespace tensorflow {
namespace serving {
namespace {

// Signature name is keyed off this in the JSON request object.
constexpr char kPredictRequestSignatureKey[] = "signature_name";

// All tensors are keyed off this in the JSON request object,
// when request format is JsonPredictRequestFormat::kRow.
constexpr char kPredictRequestInstancesKey[] = "instances";

// All tensors are keyed off this in the JSON request object,
// when request format is JsonPredictRequestFormat::kColumnar.
constexpr char kPredictRequestInputsKey[] = "inputs";

// All examples are keyed off this in the JSON request object.
constexpr char kClassifyRegressRequestContextKey[] = "context";

// All examples are keyed off this in the JSON request object.
constexpr char kClassifyRegressRequestExamplesKey[] = "examples";

// All tensors are keyed off this in the JSON response object,
// when request format is JsonPredictRequestFormat::kRow.
constexpr char kPredictResponsePredictionsKey[] = "predictions";

// All tensors are keyed off this in the JSON response object,
// when request format is JsonPredictRequestFormat::kColumnar
constexpr char kPredictResponseOutputsKey[] = "outputs";

// All classification/regression results are keyed off this
// in the JSON response object.
constexpr char kClassifyRegressResponseKey[] = "results";

// All binary (base64 encoded) strings are keyd off this in JSON.
constexpr char kBase64Key[] = "b64";

// Suffix for name of tensors that represent bytes (as opposed to strings).
constexpr char kBytesTensorNameSuffix[] = "_bytes";

using RapidJsonWriter = rapidjson::PrettyWriter<rapidjson::StringBuffer>;

string JsonTypeString(const rapidjson::Value& val) {
  switch (val.GetType()) {
    case rapidjson::kNullType:
      return "Null";
    case rapidjson::kFalseType:
      return "False";
    case rapidjson::kTrueType:
      return "True";
    case rapidjson::kObjectType:
      return "Object";
    case rapidjson::kArrayType:
      return "Array";
    case rapidjson::kStringType:
      return "String";
    case rapidjson::kNumberType:
      return "Number";
  }
}

template <typename dtype>
bool StringToDecimal(const absl::string_view s, dtype* out) {
  return absl::SimpleAtof(s, out);
}

template <>
bool StringToDecimal(const absl::string_view s, double* out) {
  return absl::SimpleAtod(s, out);
}

template <typename dtype>
bool WriteDecimal(RapidJsonWriter* writer, dtype val) {
  static_assert(
      std::is_same<dtype, float>::value || std::is_same<dtype, double>::value,
      "Only floating-point value types are supported.");
  // We do not use native writer->Double() API as float -> double conversion
  // causes noise digits to be added (due to the way floating point numbers are
  // generally represented in binary, nothing to do with the API itself). So a
  // float value of 0.2 can get written out as 0.2000000012322 (or some such).
  //
  // To get around this, we write the string representation of the float number
  // as a raw JSON value (annotated as kNumberType, to ensure JSON does not
  // quote the string).
  string decimal_str;
  if (std::isfinite(val)) {
    decimal_str = absl::StrCat(val);

    // If converted number does not roundtrip, format using full precision.
    dtype num;
    if (!StringToDecimal(decimal_str, &num)) {
      return false;
    }
    if (val != num) {
      decimal_str = absl::StrFormat(
          "%.*g", std::numeric_limits<dtype>::max_digits10, val);
    }

    // Add trailing '.0' for whole numbers and those not in scientific notation.
    // StrCat() formats numbers in six-digit (printf "%g"), numbers like 9000000
    // and .00003 get written as 9e+06 and 3e-05 (scientific notation).
    //
    // Not adding '.0' can lead to lists containing mix of decimal and whole
    // numbers -- making it difficult for consumers to pick the correct type to
    // store these numbers (note, JSON does not have metadata to describe types.
    // These are inferred from the tokens).
    if (decimal_str.find('.') == string::npos &&
        decimal_str.find('e') == string::npos) {
      absl::StrAppend(&decimal_str, ".0");
    }
  } else if (std::isnan(val)) {
    decimal_str = "NaN";
  } else if (std::isinf(val)) {
    decimal_str = std::signbit(val) ? "-Infinity" : "Infinity";
  }
  return writer->RawValue(decimal_str.c_str(), decimal_str.size(),
                          rapidjson::kNumberType);
}

// Stringify JSON value (only for use in error reporting or debugging).
string JsonValueToString(const rapidjson::Value& val) {
  // TODO(b/67042542): Truncate large values.
  rapidjson::StringBuffer buffer;
  RapidJsonWriter writer(buffer);
  // Write decimal numbers explicitly so inf/nan's get printed
  // correctly. RapidJsonWriter() does not write these correctly.
  if (val.IsFloat()) {
    WriteDecimal(&writer, val.GetFloat());
  } else if (val.IsDouble()) {
    WriteDecimal(&writer, val.GetDouble());
  } else {
    val.Accept(writer);
  }
  return buffer.GetString();
}

// Prints a shape array in [x, y, z] format.
string ShapeToString(const TensorShapeProto& shape) {
  return shape.unknown_rank()
             ? "[?]"
             : absl::StrCat(
                   "[",
                   absl::StrJoin(
                       shape.dim(), ",",
                       [](string* out, const TensorShapeProto_Dim& dim) {
                         out->append(absl::StrCat(dim.size()));
                       }),
                   "]");
}

bool IsShapeEqual(const TensorShapeProto& lhs, const TensorShapeProto& rhs) {
  return !lhs.unknown_rank() && !rhs.unknown_rank() &&
         lhs.dim_size() == rhs.dim_size() &&
         std::equal(lhs.dim().begin(), lhs.dim().end(), rhs.dim().begin(),
                    [](const TensorShapeProto_Dim& lhs,
                       const TensorShapeProto_Dim& rhs) {
                      return lhs.size() == rhs.size();
                    });
}

Status TypeError(const rapidjson::Value& val, DataType dtype) {
  return errors::InvalidArgument(
      "JSON Value: ", JsonValueToString(val), " Type: ", JsonTypeString(val),
      " is not of expected type: ", DataTypeString(dtype));
}

Status Base64FormatError(const rapidjson::Value& val) {
  return errors::InvalidArgument("JSON Value: ", JsonValueToString(val),
                                 " not formatted correctly for base64 data");
}

template <typename... Args>
Status FormatError(const rapidjson::Value& val, Args&&... args) {
  return errors::InvalidArgument("JSON Value: ",
    JsonValueToString(val), " ", std::forward<Args>(args)...);
}

Status FormatSignatureError(const rapidjson::Value& val) {
  return errors::InvalidArgument(
      "JSON Value: ", JsonValueToString(val),
      " not formatted correctly. 'signature_name' key must be a string value.");
}

Status LossyDecimalError(const rapidjson::Value& val, const string& target) {
  return errors::InvalidArgument(
      "Cannot convert JSON value: ", JsonValueToString(val), " to ", target,
      " without loss of precision.");
}

template <typename T>
bool IsLosslessDecimal(const rapidjson::Value& val) {
  static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
                "Only floating point value types are supported.");
  static_assert(std::numeric_limits<T>::radix == 2,
                "Floating point type must have base 2.");

  // Note, we use GetDouble() for both types as std::isfinite() returns false
  // for decimal values that do not fit in float (due to the static_cast<> used
  // in converting double to float in GetFloat() call).
  if (!std::isfinite(val.GetDouble())) return true;

  // Maximum integer value that can be represented by the floating type.
  static constexpr int64 kMaxInt = (1LL << std::numeric_limits<T>::digits) - 1;

  if (val.IsUint64()) {
    return val.GetUint64() <= kMaxInt;
  }
  if (val.IsInt64()) {
    return std::abs(val.GetInt64()) <= kMaxInt;
  }
  return val.GetDouble() <= std::numeric_limits<T>::max() &&
         val.GetDouble() >= std::numeric_limits<T>::lowest();
}

// Adds a JSON value to Tensor. Returns error if value cannot be converted
// to dtype. In case of error (output) tensor is not modified.
//
// TODO(b/67042542): Templatize on DataType
Status AddValueToTensor(const rapidjson::Value& val, DataType dtype,
                        TensorProto* tensor) {
  switch (dtype) {
    case DT_FLOAT:
      if (!val.IsNumber()) return TypeError(val, dtype);
      if (!IsLosslessDecimal<float>(val)) {
        return LossyDecimalError(val, "float");
      }
      tensor->add_float_val(val.GetFloat());
      break;

    case DT_DOUBLE:
      if (!val.IsNumber()) return TypeError(val, dtype);
      if (!IsLosslessDecimal<double>(val)) {
        return LossyDecimalError(val, "double");
      }
      tensor->add_double_val(val.GetDouble());
      break;

    case DT_INT32:
    case DT_INT16:
    case DT_INT8:
    case DT_UINT8:
      if (!val.IsInt()) return TypeError(val, dtype);
      tensor->add_int_val(val.GetInt());
      break;

    case DT_STRING:
      if (!val.IsString()) return TypeError(val, dtype);
      tensor->add_string_val(val.GetString(), val.GetStringLength());
      break;

    case DT_INT64:
      if (!val.IsInt64()) return TypeError(val, dtype);
      tensor->add_int64_val(val.GetInt64());
      break;

    case DT_BOOL:
      if (!val.IsBool()) return TypeError(val, dtype);
      tensor->add_bool_val(val.GetBool());
      break;

    case DT_UINT32:
      if (!val.IsUint()) return TypeError(val, dtype);
      tensor->add_uint32_val(val.GetUint());
      break;

    case DT_UINT64:
      if (!val.IsUint64()) return TypeError(val, dtype);
      tensor->add_uint64_val(val.GetUint64());
      break;

    default:
      return errors::Unimplemented(
          "Conversion of JSON Value: ", JsonValueToString(val),
          " to type: ", DataTypeString(dtype));
  }
  return Status::OK();
}

// Computes and fills TensorShape corresponding to a JSON value.
//
// `val` can be scalar or list or list of lists with arbitrary nesting. If a
// scalar (non array) is passed, we do not add dimension info to shape (as
// scalars do not have a dimension).
void GetDenseTensorShape(const rapidjson::Value& val, TensorShapeProto* shape) {
  if (!val.IsArray()) return;
  const auto size = val.Size();
  shape->add_dim()->set_size(size);
  if (size > 0) {
    GetDenseTensorShape(val[0], shape);
  }
}

// Returns size (number of elements) expected in a tensor of given shape.
int GetTensorSize(const TensorShapeProto& shape) {
  int size = 1;
  for (const auto& dim : shape.dim()) {
    size *= dim.size();
  }
  return size;
}

bool IsValBase64Object(const rapidjson::Value& val) {
  // Base64 encoded data is a JSON object formatted as:
  // { "b64" : "<base64 encoded string" }
  // Note:
  // - The object must contain only one key named as 'b64'
  // - The key must map to a JSON string.
  if (val.IsObject()) {
    const auto itr = val.FindMember(kBase64Key);
    if (itr != val.MemberEnd() && val.MemberCount() == 1 &&
        itr->value.IsString()) {
      return true;
    }
  }
  return false;
}

// Decodes a base64 encoded JSON string. Such strings are expected to be
// expressed as JSON object with only one (string) key as "b64" and value of
// this key a base64 encoded string.
Status JsonDecodeBase64Object(const rapidjson::Value& val,
                              string* decoded_val) {
  if (!IsValBase64Object(val)) {
    return Base64FormatError(val);
  }
  const auto itr = val.FindMember(kBase64Key);
  if (!absl::Base64Unescape(absl::string_view(itr->value.GetString(),
                                              itr->value.GetStringLength()),
                            decoded_val)) {
    return errors::InvalidArgument("Unable to base64 decode");
  }
  return Status::OK();
}

// Fills tensor values.
Status FillTensorProto(const rapidjson::Value& val, int level, DataType dtype,
                       int* val_count, TensorProto* tensor) {
  const auto rank = tensor->tensor_shape().dim_size();
  if (!val.IsArray()) {
    // DOM tree for a (dense) tensor will always have all values
    // at same (leaf) level equal to the rank of the tensor.
    if (level != rank) {
      return errors::InvalidArgument(
          "JSON Value: ", JsonValueToString(val),
          " found at incorrect level: ", level + 1,
          " in the JSON DOM. Expected at level: ", rank);
    }
    Status status;
    if (val.IsObject()) {
      status = (dtype == DT_STRING)
                   ? JsonDecodeBase64Object(val, tensor->add_string_val())
                   : TypeError(val, dtype);
    } else {
      status = AddValueToTensor(val, dtype, tensor);
    }
    if (status.ok()) (*val_count)++;
    return status;
  }

  // If list is nested deeper than rank, stop processing.
  if (level >= rank) {
    return errors::InvalidArgument(
        "Encountered list at unexpected level: ", level, " expected < ", rank);
  }

  // Ensure list is of expected size for our level.
  if (val.Size() != tensor->tensor_shape().dim(level).size()) {
    return errors::InvalidArgument(
        "Encountered list at unexpected size: ", val.Size(),
        " at level: ", level,
        " expected size: ", tensor->tensor_shape().dim(level).size());
  }

  // All OK, recurse into elements of the list.
  for (const auto& v : val.GetArray()) {
    TF_RETURN_IF_ERROR(FillTensorProto(v, level + 1, dtype, val_count, tensor));
  }

  return Status::OK();
}

// Converts a JSON value to tensor and add it to tensor_map.
//
// 'name' is the alias/name of the tensor as it appears in signature def.
// size/shape/tensor_map are all maps keyed by the name of the tensor and
// get updated as part of the add operation.
Status AddInstanceItem(const rapidjson::Value& item, const string& name,
                       const ::google::protobuf::Map<string, TensorInfo>& tensorinfo_map,
                       ::google::protobuf::Map<string, int>* size_map,
                       ::google::protobuf::Map<string, TensorShapeProto>* shape_map,
                       ::google::protobuf::Map<string, TensorProto>* tensor_map) {
  if (!tensorinfo_map.count(name)) {
    return errors::InvalidArgument("JSON object: does not have named input: ",
                                   name);
  }
  int size = 0;
  const auto dtype = tensorinfo_map.at(name).dtype();
  auto* tensor = &(*tensor_map)[name];
  tensor->mutable_tensor_shape()->Clear();
  GetDenseTensorShape(item, tensor->mutable_tensor_shape());
  TF_RETURN_IF_ERROR(
      FillTensorProto(item, 0 /* level */, dtype, &size, tensor));
  if (!size_map->count(name)) {
    (*size_map)[name] = size;
    (*shape_map)[name] = tensor->tensor_shape();
  } else if ((*size_map)[name] != size) {
    return errors::InvalidArgument("Expecting tensor size: ", (*size_map)[name],
                                   " but got: ", size);
  } else if (!IsShapeEqual((*shape_map)[name], tensor->tensor_shape())) {
    return errors::InvalidArgument(
        "Expecting shape ", ShapeToString((*shape_map)[name]),
        " but got: ", ShapeToString(tensor->tensor_shape()));
  }
  return Status::OK();
}

Status ParseJson(const absl::string_view json, rapidjson::Document* doc) {
  if (json.empty()) {
    return errors::InvalidArgument("JSON Parse error: The document is empty");
  }

  // `json` may not be null-terminated (read from a mem buffer).
  // Wrap it in an input stream before attempting to Parse().
  rapidjson::MemoryStream ms(json.data(), json.size());
  rapidjson::EncodedInputStream<rapidjson::UTF8<>, rapidjson::MemoryStream>
      jsonstream(ms);
  // TODO(b/67042542): Switch to using custom stack for parsing to protect
  // against deep nested structures causing excessive recursion/SO.
  if (doc->ParseStream<rapidjson::kParseNanAndInfFlag>(jsonstream)
          .HasParseError()) {
    return errors::InvalidArgument(
        "JSON Parse error: ", rapidjson::GetParseError_En(doc->GetParseError()),
        " at offset: ", doc->GetErrorOffset());
  }

  // We expect top level JSON to be an object.
  // {
  //   "signature_name" : <string>
  //   "instances" : [ <atlest one element> ]
  //   ...
  // }
  if (!doc->IsObject()) {
    return FormatError(*doc, "Is not object");
  }
  return Status::OK();
}

template <typename RequestTypeProto>
Status FillSignature(const rapidjson::Document& doc,
                     RequestTypeProto* request) {
  // Fill in (optional) signature_name.
  auto itr = doc.FindMember(kPredictRequestSignatureKey);
  if (itr != doc.MemberEnd()) {
    if (!itr->value.IsString()) {
      return FormatSignatureError(doc);
    }
    request->mutable_model_spec()->set_signature_name(
        itr->value.GetString(), itr->value.GetStringLength());
  }
  return Status::OK();
}

Status FillTensorMapFromInstancesList(
    const rapidjson::Value::MemberIterator& itr,
    const ::google::protobuf::Map<string, tensorflow::TensorInfo>& tensorinfo_map,
    ::google::protobuf::Map<string, TensorProto>* tensor_map) {
  // "instances" array can either be a plain list or list of objects (for named
  // tensors) but not a mix of both. Each object must have one key for each
  // named tensor.
  if (!itr->value[0].IsObject() && tensorinfo_map.size() > 1) {
    return errors::InvalidArgument(
        "instances is a plain list, but expecting list of objects as multiple "
        "input tensors required as per tensorinfo_map");
  }

  auto IsElementObject = [](const rapidjson::Value& val) {
    return val.IsObject() && !IsValBase64Object(val);
  };

  const bool elements_are_objects = IsElementObject(itr->value[0]);

  std::set<string> input_names;
  for (const auto& kv : tensorinfo_map) input_names.insert(kv.first);

  // Add each element of "instances" array to tensor.
  //
  // Each element must yield one tensor of same shape and size. All elements get
  // batched into one tensor with the first dimension equal to the number of
  // elements in the instances array.
  tensor_map->clear();
  ::google::protobuf::Map<string, int> size_map;
  ::google::protobuf::Map<string, TensorShapeProto> shape_map;
  int tensor_count = 0;
  for (const auto& elem : itr->value.GetArray()) {
    if (elements_are_objects) {
      if (!IsElementObject(elem)) {
        return errors::InvalidArgument("Expecting object but got list at item ",
                                       tensor_count, " of input list");
      }
      std::set<string> object_keys;
      for (const auto& kv : elem.GetObject()) {
        const string& name = kv.name.GetString();
        object_keys.insert(name);
        const auto status = AddInstanceItem(kv.value, name, tensorinfo_map,
                                            &size_map, &shape_map, tensor_map);
        if (!status.ok()) {
          return errors::InvalidArgument(
              "Failed to process element: ", tensor_count, " key: ", name,
              " of 'instances' list. Error: ", status.ToString());
        }
      }
      if (input_names != object_keys) {
        return errors::InvalidArgument(
            "Failed to process element: ", tensor_count,
            " of 'instances' list. JSON object: ", JsonValueToString(elem),
            " must only have keys: ", absl::StrJoin(input_names, ","));
      }
    } else {
      if (IsElementObject(elem)) {
        return errors::InvalidArgument(
            "Expecting value/list but got object at item ", tensor_count,
            " of input list");
      }
      const auto& name = tensorinfo_map.begin()->first;
      const auto status = AddInstanceItem(elem, name, tensorinfo_map, &size_map,
                                          &shape_map, tensor_map);
      if (!status.ok()) {
        return errors::InvalidArgument(
            "Failed to process element: ", tensor_count,
            " of 'instances' list. Error: ", status.ToString());
      }
    }
    tensor_count++;
  }

  // Now that all individual tensors from "instances" array are added,
  // fix the resulting shape of the final tensor.
  for (auto& kv : *tensor_map) {
    const string& name = kv.first;
    auto* tensor = &kv.second;
    tensor->set_dtype(tensorinfo_map.at(name).dtype());
    const auto& shape = shape_map.at(name);
    auto* output_shape = tensor->mutable_tensor_shape();
    output_shape->Clear();
    output_shape->add_dim()->set_size(tensor_count);
    for (const auto& d : shape.dim())
      output_shape->add_dim()->set_size(d.size());
  }

  return Status::OK();
}

Status FillTensorMapFromInputsMap(
    const rapidjson::Value::MemberIterator& itr,
    const ::google::protobuf::Map<string, tensorflow::TensorInfo>& tensorinfo_map,
    ::google::protobuf::Map<string, TensorProto>* tensor_map) {
  // "inputs" key can hold a value that is one of the following:
  // - a list or base64 object (when there is only one named input)
  // - a object of key->value pairs (when there are multiple named inputs)
  const rapidjson::Value& val = itr->value;
  if (!val.IsObject() || IsValBase64Object(val)) {
    if (tensorinfo_map.size() > 1) {
      return errors::InvalidArgument(
          "inputs is a plain value/list, but expecting an object as multiple "
          "input tensors required as per tensorinfo_map");
    }

    auto* tensor = &(*tensor_map)[tensorinfo_map.begin()->first];
    tensor->set_dtype(tensorinfo_map.begin()->second.dtype());
    GetDenseTensorShape(val, tensor->mutable_tensor_shape());
    int unused_size = 0;
    TF_RETURN_IF_ERROR(FillTensorProto(val, 0 /* level */, tensor->dtype(),
                                       &unused_size, tensor));
  } else {
    for (const auto& kv : tensorinfo_map) {
      const auto& name = kv.first;
      auto item = val.FindMember(name.c_str());
      if (item == val.MemberEnd()) {
        return errors::InvalidArgument("Missing named input: ", name,
                                       " in 'inputs' object.");
      }
      const auto dtype = kv.second.dtype();
      auto* tensor = &(*tensor_map)[name];
      tensor->set_dtype(dtype);
      tensor->mutable_tensor_shape()->Clear();
      GetDenseTensorShape(item->value, tensor->mutable_tensor_shape());
      int unused_size = 0;
      TF_RETURN_IF_ERROR(FillTensorProto(item->value, 0 /* level */, dtype,
                                         &unused_size, tensor));
    }
  }
  return Status::OK();
}

}  // namespace

Status FillPredictRequestFromJson(
    const absl::string_view json,
    const std::function<tensorflow::Status(
        const string&, ::google::protobuf::Map<string, tensorflow::TensorInfo>*)>&
        get_tensorinfo_map,
    PredictRequest* request, JsonPredictRequestFormat* format) {
  rapidjson::Document doc;
  *format = JsonPredictRequestFormat::kInvalid;
  TF_RETURN_IF_ERROR(ParseJson(json, &doc));
  TF_RETURN_IF_ERROR(FillSignature(doc, request));

  ::google::protobuf::Map<string, tensorflow::TensorInfo> tensorinfo_map;
  const string& signame = request->model_spec().signature_name();
  TF_RETURN_IF_ERROR(get_tensorinfo_map(signame, &tensorinfo_map));
  if (tensorinfo_map.empty()) {
    return errors::InvalidArgument("Failed to get input map for signature: ",
                                   signame.empty() ? "DEFAULT" : signame);
  }

  //
  // Fill in tensors from either "instances" or "inputs" key.
  //
  auto itr_instances = doc.FindMember(kPredictRequestInstancesKey);
  auto itr_inputs = doc.FindMember(kPredictRequestInputsKey);
  if (itr_instances != doc.MemberEnd()) {
    if (itr_inputs != doc.MemberEnd()) {
      return FormatError(doc, "Not formatted correctly expecting only",
        " one of '", kPredictRequestInputsKey, "' or '",
        kPredictRequestInstancesKey, "' keys to exist ");
    }
    if (!itr_instances->value.IsArray()) {
      return FormatError(doc, "Excepting '",
        kPredictRequestInstancesKey, "' to be an list/array");
    }
    if (!itr_instances->value.Capacity()) {
      return FormatError(doc, "No values in '",
        kPredictRequestInstancesKey, "' array");
    }
    *format = JsonPredictRequestFormat::kRow;
    return FillTensorMapFromInstancesList(itr_instances, tensorinfo_map,
                                          request->mutable_inputs());
  } else if (itr_inputs != doc.MemberEnd()) {
    if (itr_instances != doc.MemberEnd()) {
      return FormatError(doc, "Not formatted correctly expecting only",
        " one of '", kPredictRequestInputsKey, "' or '",
        kPredictRequestInstancesKey, "' keys to exist ");
    }
    *format = JsonPredictRequestFormat::kColumnar;
    return FillTensorMapFromInputsMap(itr_inputs, tensorinfo_map,
                                      request->mutable_inputs());
  }
  return errors::InvalidArgument("Missing 'inputs' or 'instances' key");
}

namespace {

bool IsFeatureOfKind(const Feature& feature, Feature::KindCase kind) {
  return feature.kind_case() == Feature::KIND_NOT_SET ||
         feature.kind_case() == kind;
}

Status IncompatibleFeatureKindError(const string& feature_name,
                                    const Feature& feature) {
  string kind_str;
  switch (feature.kind_case()) {
    case Feature::KindCase::kBytesList:
      kind_str = "bytes";
      break;
    case Feature::KindCase::kFloatList:
      kind_str = "float";
      break;
    case Feature::KindCase::kInt64List:
      kind_str = "int64";
      break;
    case Feature::KindCase::KIND_NOT_SET:
      kind_str = "UNKNOWN";
      break;
  }
  return errors::InvalidArgument("Unexpected element type in feature: ",
                                 feature_name, " expecting type: ", kind_str);
}

// Adds a JSON value to Feature proto. Returns error if value cannot be
// converted to dtype. In case of error (output) tensor is not modified.
Status AddValueToFeature(const rapidjson::Value& val,
                         const string& feature_name, Feature* feature) {
  switch (val.GetType()) {
    case rapidjson::kNullType:
      return errors::InvalidArgument(
          "Feature: ", feature_name,
          " has element with unexpected JSON type: ", JsonTypeString(val));
    case rapidjson::kFalseType:
    case rapidjson::kTrueType:
      if (!IsFeatureOfKind(*feature, Feature::KindCase::kInt64List)) {
        return IncompatibleFeatureKindError(feature_name, *feature);
      }
      feature->mutable_int64_list()->add_value(val.GetBool() ? 1 : 0);
      break;
    case rapidjson::kObjectType:
      if (!IsValBase64Object(val)) {
        return errors::InvalidArgument(
            "Feature: ", feature_name,
            " has element with unexpected JSON type: ", JsonTypeString(val));
      }
      if (!IsFeatureOfKind(*feature, Feature::KindCase::kBytesList)) {
        return IncompatibleFeatureKindError(feature_name, *feature);
      }
      TF_RETURN_IF_ERROR(JsonDecodeBase64Object(
          val, feature->mutable_bytes_list()->add_value()));
      break;
    case rapidjson::kArrayType:
      return errors::InvalidArgument(
          "Feature: ", feature_name,
          " has element with unexpected JSON type: ", JsonTypeString(val));
    case rapidjson::kStringType:
      if (!IsFeatureOfKind(*feature, Feature::KindCase::kBytesList)) {
        return IncompatibleFeatureKindError(feature_name, *feature);
      }
      feature->mutable_bytes_list()->add_value(val.GetString(),
                                               val.GetStringLength());
      break;
    case rapidjson::kNumberType:
      if (val.IsDouble()) {
        if (!IsFeatureOfKind(*feature, Feature::KindCase::kFloatList)) {
          return IncompatibleFeatureKindError(feature_name, *feature);
        }
        if (!IsLosslessDecimal<float>(val)) {
          return LossyDecimalError(val, "float");
        }
        feature->mutable_float_list()->add_value(val.GetFloat());
      } else {
        if (!IsFeatureOfKind(*feature, Feature::KindCase::kInt64List)) {
          return IncompatibleFeatureKindError(feature_name, *feature);
        }
        if (!val.IsInt64() && val.IsUint64()) {
          return errors::InvalidArgument(
              "Feature: ", feature_name,
              " has uint64 element. Only int64 is supported.");
        }
        feature->mutable_int64_list()->add_value(val.GetInt64());
      }
      break;
  }
  return Status::OK();
}

Status MakeExampleFromJsonObject(const rapidjson::Value& val,
                                 Example* example) {
  if (!val.IsObject()) {
    return errors::InvalidArgument("Example must be JSON object but got JSON ",
                                   JsonTypeString(val));
  }
  for (const auto& kv : val.GetObject()) {
    const string& name = kv.name.GetString();
    const auto& content = kv.value;
    Feature feature;
    if (content.IsArray()) {
      for (const auto& val : content.GetArray()) {
        TF_RETURN_IF_ERROR(AddValueToFeature(val, name, &feature));
      }
    } else {
      TF_RETURN_IF_ERROR(AddValueToFeature(content, name, &feature));
    }
    (*example->mutable_features()->mutable_feature())[name] = feature;
  }
  return Status::OK();
}

template <typename RequestProto>
Status FillClassifyRegressRequestFromJson(const absl::string_view json,
                                          RequestProto* request) {
  rapidjson::Document doc;
  TF_RETURN_IF_ERROR(ParseJson(json, &doc));
  TF_RETURN_IF_ERROR(FillSignature(doc, request));

  // Fill in (optional) Example context.
  bool has_context = false;
  auto* const input = request->mutable_input();
  auto itr = doc.FindMember(kClassifyRegressRequestContextKey);
  if (itr != doc.MemberEnd()) {
    TF_RETURN_IF_ERROR(MakeExampleFromJsonObject(
        itr->value,
        input->mutable_example_list_with_context()->mutable_context()));
    has_context = true;
  }

  // Fill in list of Examples.
  itr = doc.FindMember(kClassifyRegressRequestExamplesKey);
  if (itr == doc.MemberEnd()) {
    return FormatError(doc, "When method is classify, key '",
      kClassifyRegressRequestExamplesKey,
      "' is expected and was not found");
  }
  if (!itr->value.IsArray()) {
    return FormatError(doc, "Expecting '",
      kClassifyRegressRequestExamplesKey,
      "' value to be an list/array");
  }
  if (!itr->value.Capacity()) {
    return FormatError(doc, "'", kClassifyRegressRequestExamplesKey,
      "' value is an empty array");
  }
  for (const auto& val : itr->value.GetArray()) {
    TF_RETURN_IF_ERROR(MakeExampleFromJsonObject(
        val, has_context
                 ? input->mutable_example_list_with_context()->add_examples()
                 : input->mutable_example_list()->add_examples()));
  }

  return Status::OK();
}

}  // namespace

Status FillClassificationRequestFromJson(const absl::string_view json,
                                         ClassificationRequest* request) {
  return FillClassifyRegressRequestFromJson(json, request);
}

Status FillRegressionRequestFromJson(const absl::string_view json,
                                     RegressionRequest* request) {
  return FillClassifyRegressRequestFromJson(json, request);
}

namespace {


bool IsNamedTensorBytes(const string& name, const TensorProto& tensor) {
  // TODO(b/67042542): Is DT_STRING the only way to represent bytes?
  //
  // TODO(b/67042542): Add additional metadata in signaturedef to not rely on
  // the name being formed with a suffix (this is borrowed from CMLE, for
  // consistency, but is sub-optimal and TF serving should do better.
  return tensor.dtype() == DT_STRING &&
         absl::EndsWith(name, kBytesTensorNameSuffix);
}


Status AddSingleValueAndAdvance(const TensorProto& tensor, bool string_as_bytes,
                                RapidJsonWriter* writer, int* offset) {
  bool success = false;
  switch (tensor.dtype()) {
    case DT_FLOAT:
      success = WriteDecimal(writer, tensor.float_val(*offset));
      break;

    case DT_DOUBLE:
      success = WriteDecimal(writer, tensor.double_val(*offset));
      break;

    case DT_INT32:
    case DT_INT16:
    case DT_INT8:
    case DT_UINT8:
      success = writer->Int(tensor.int_val(*offset));
      break;

    case DT_STRING: {
      const string& str = tensor.string_val(*offset);
      if (string_as_bytes) {
        // Write bytes as { "b64": <base64-encoded-string> }
        string base64;
        absl::Base64Escape(str, &base64);
        writer->StartObject();
        writer->Key(kBase64Key);
        success = writer->String(base64.c_str(), base64.size());
        writer->EndObject();
      } else {
        success = writer->String(str.c_str(), str.size());
      }
      break;
    }

    case DT_INT64:
      success = writer->Int64(tensor.int64_val(*offset));
      break;

    case DT_BOOL:
      success = writer->Bool(tensor.bool_val(*offset));
      break;

    case DT_UINT32:
      success = writer->Uint(tensor.uint32_val(*offset));
      break;

    case DT_UINT64:
      success = writer->Uint64(tensor.uint64_val(*offset));
      break;

    default:
      success = false;
      break;
  }
  if (!success) {
    return errors::InvalidArgument(
        "Failed to write JSON value for tensor type: ",
        DataTypeString(tensor.dtype()));
  }
  (*offset)++;
  return Status::OK();
}

Status AddTensorValues(const TensorProto& tensor, bool string_as_bytes, int dim,
                       RapidJsonWriter* writer, int* offset) {
  // Scalar values dont need to be enclosed in an array.
  // Just write the (single) value out and return.
  if (dim > tensor.tensor_shape().dim_size() - 1) {
    return AddSingleValueAndAdvance(tensor, string_as_bytes, writer, offset);
  }
  writer->StartArray();
  if (dim == tensor.tensor_shape().dim_size() - 1) {
    for (int i = 0; i < tensor.tensor_shape().dim(dim).size(); i++) {
      TF_RETURN_IF_ERROR(
          AddSingleValueAndAdvance(tensor, string_as_bytes, writer, offset));
    }
  } else {
    for (int i = 0; i < tensor.tensor_shape().dim(dim).size(); i++) {
      TF_RETURN_IF_ERROR(
          AddTensorValues(tensor, string_as_bytes, dim + 1, writer, offset));
    }
  }
  writer->EndArray();
  return Status::OK();
}

Status MakeRowFormatJsonFromTensors(
    const ::google::protobuf::Map<string, TensorProto>& tensor_map, string* json) {
  // Verify if each named tensor has same first dimension. The first dimension
  // is the batchsize and for an output to be consistent, all named tensors must
  // be batched to the same size.
  int batch_size = 0;
  // Track offset into value list for each tensor (used in the next section).
  std::unordered_map<string, int> offset_map;
  for (const auto& kv : tensor_map) {
    const auto& name = kv.first;
    const auto& tensor = kv.second;
    if (tensor.tensor_shape().dim_size() == 0) {
      return errors::InvalidArgument("Tensor name: ", name,
                                     " has no shape information ");
    }
    const int cur_batch_size = tensor.tensor_shape().dim(0).size();
    if (cur_batch_size < 1) {
      return errors::InvalidArgument(
          "Tensor name: ", name, " has invalid batch size: ", cur_batch_size);
    }
    if (batch_size != 0 && batch_size != cur_batch_size) {
      return errors::InvalidArgument(
          "Tensor name: ", name,
          " has inconsistent batch size: ", cur_batch_size,
          " expecting: ", batch_size);
    }
    batch_size = cur_batch_size;
    offset_map.insert({name, 0});
  }

  rapidjson::StringBuffer buffer;
  rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
  writer.StartObject();
  writer.Key(kPredictResponsePredictionsKey);
  writer.StartArray();
  const bool elements_are_objects = tensor_map.size() > 1;
  for (int item = 0; item < batch_size; item++) {
    if (elements_are_objects) writer.StartObject();
    writer.SetFormatOptions(rapidjson::kFormatSingleLineArray);
    for (const auto& kv : tensor_map) {
      const auto& name = kv.first;
      const auto& tensor = kv.second;
      if (elements_are_objects) writer.Key(name.c_str());
      TF_RETURN_IF_ERROR(AddTensorValues(
          tensor, IsNamedTensorBytes(name, tensor),
          1 /* dimension, we start from 1st as 0th is batch dimension */,
          &writer, &offset_map.at(name)));
    }
    writer.SetFormatOptions(rapidjson::kFormatDefault);
    if (elements_are_objects) writer.EndObject();
  }
  writer.EndArray();
  writer.EndObject();
  json->assign(buffer.GetString());
  return Status::OK();
}

Status MakeColumnarFormatJsonFromTensors(
    const ::google::protobuf::Map<string, TensorProto>& tensor_map, string* json) {
  rapidjson::StringBuffer buffer;
  rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
  writer.StartObject();
  writer.Key(kPredictResponseOutputsKey);
  const bool elements_are_objects = tensor_map.size() > 1;
  if (elements_are_objects) writer.StartObject();
  for (const auto& kv : tensor_map) {
    const auto& name = kv.first;
    const auto& tensor = kv.second;
    if (elements_are_objects) writer.Key(name.c_str());
    int unused_offset = 0;
    TF_RETURN_IF_ERROR(AddTensorValues(tensor, IsNamedTensorBytes(name, tensor),
                                       0, &writer, &unused_offset));
  }
  if (elements_are_objects) writer.EndObject();
  writer.EndObject();
  json->assign(buffer.GetString());
  return Status::OK();
}

}  // namespace

Status MakeJsonFromTensors(const ::google::protobuf::Map<string, TensorProto>& tensor_map,
                           JsonPredictRequestFormat format, string* json) {
  if (tensor_map.empty()) {
    return errors::InvalidArgument("Cannot convert empty tensor map to JSON");
  }

  switch (format) {
    case JsonPredictRequestFormat::kInvalid:
      return errors::InvalidArgument("Invalid request format");
    case JsonPredictRequestFormat::kRow:
      return MakeRowFormatJsonFromTensors(tensor_map, json);
    case JsonPredictRequestFormat::kColumnar:
      return MakeColumnarFormatJsonFromTensors(tensor_map, json);
  }
}

Status MakeJsonFromClassificationResult(const ClassificationResult& result,
                                        string* json) {
  if (result.classifications_size() == 0) {
    return errors::InvalidArgument(
        "Cannot convert empty ClassificationResults to JSON");
  }

  rapidjson::StringBuffer buffer;
  rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
  writer.StartObject();
  writer.Key(kClassifyRegressResponseKey);
  writer.StartArray();
  for (const auto& classifications : result.classifications()) {
    writer.SetFormatOptions(rapidjson::kFormatSingleLineArray);
    writer.StartArray();
    for (const auto& elem : classifications.classes()) {
      writer.StartArray();
      if (!writer.String(elem.label().c_str(), elem.label().size())) {
        return errors::Internal("Failed to write class label: ", elem.label(),
                                " to output JSON buffer");
      }
      if (!WriteDecimal(&writer, elem.score())) {
        return errors::Internal("Failed to write class score : ", elem.score(),
                                " to output JSON buffer");
      }
      writer.EndArray();
    }
    writer.EndArray();
    writer.SetFormatOptions(rapidjson::kFormatDefault);
  }
  writer.EndArray();
  writer.EndObject();
  json->assign(buffer.GetString());
  return Status::OK();
}

Status MakeJsonFromRegressionResult(const RegressionResult& result,
                                    string* json) {
  if (result.regressions_size() == 0) {
    return errors::InvalidArgument(
        "Cannot convert empty RegressionResults to JSON");
  }

  rapidjson::StringBuffer buffer;
  rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
  writer.StartObject();
  writer.Key(kClassifyRegressResponseKey);
  writer.SetFormatOptions(rapidjson::kFormatSingleLineArray);
  writer.StartArray();
  for (const auto& regression : result.regressions()) {
    if (!WriteDecimal(&writer, regression.value())) {
      return errors::Internal("Failed to write regression value : ",
                              regression.value(), " to output JSON buffer");
    }
  }
  writer.EndArray();
  writer.EndObject();
  json->assign(buffer.GetString());
  return Status::OK();
}

}  // namespace serving
}  // namespace tensorflow
