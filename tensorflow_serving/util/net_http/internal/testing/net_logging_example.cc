/* Copyright 2019 Google Inc. All Rights Reserved.

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

#include <cstddef>
#include <iostream>

#include "tensorflow_serving/util/net_http/internal/net_logging.h"

int main(int argc, char** argv) {
  NET_LOG(INFO, "started!");

  size_t size = 100;
  NET_LOG(ERROR, "read less than specified bytes : %zu", size);

  const char* url = "/url";
  NET_LOG(WARNING, "%s: read less than specified bytes : %zu", url, size);

  NET_LOG(FATAL, "aborted!");

  return 0;  // unexpected
}
