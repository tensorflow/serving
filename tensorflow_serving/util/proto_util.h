/* Copyright 2022 Google Inc. All Rights Reserved.

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
#ifndef THIRD_PARTY_TENSORFLOW_SERVING_UTIL_PROTO_UTIL_H_
#define THIRD_PARTY_TENSORFLOW_SERVING_UTIL_PROTO_UTIL_H_

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace serving {

template <typename ProtoType>
tensorflow::Status ParseProtoTextFile(const string& file, ProtoType* proto) {
  std::unique_ptr<tensorflow::ReadOnlyMemoryRegion> file_data;
  TF_RETURN_IF_ERROR(
      tensorflow::Env::Default()->NewReadOnlyMemoryRegionFromFile(file,
                                                                  &file_data));
  string file_data_str(static_cast<const char*>(file_data->data()),
                       file_data->length());
  if (tensorflow::protobuf::TextFormat::ParseFromString(file_data_str, proto)) {
    return tensorflow::OkStatus();
  } else {
    return tensorflow::errors::InvalidArgument("Invalid protobuf file: '", file,
                                               "'");
  }
}

}  // namespace serving
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_SERVING_UTIL_PROTO_UTIL_H_
