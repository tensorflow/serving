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

#include "tensorflow_serving/util/status_util.h"

namespace tensorflow {
namespace serving {

StatusProto ToStatusProto(const Status& status) {
  StatusProto status_proto;
  status_proto.set_error_code(status.code());
  if (!status.ok()) {
    status_proto.set_error_message(status.error_message());
  }
  return status_proto;
}

Status FromStatusProto(const StatusProto& status_proto) {
  return status_proto.error_code() == tensorflow::error::OK
             ? Status()
             : Status(status_proto.error_code(), status_proto.error_message());
}

}  // namespace serving
}  // namespace tensorflow
