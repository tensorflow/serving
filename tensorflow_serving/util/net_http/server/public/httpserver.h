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

// The entry point to access different HTTP server implementations.

#ifndef TENSORFLOW_SERVING_UTIL_NET_HTTP_SERVER_PUBLIC_HTTPSERVER_H_
#define TENSORFLOW_SERVING_UTIL_NET_HTTP_SERVER_PUBLIC_HTTPSERVER_H_

#include <memory>

#include "absl/memory/memory.h"

#include "tensorflow_serving/util/net_http/server/internal/evhttp_server.h"
#include "tensorflow_serving/util/net_http/server/public/httpserver_interface.h"

namespace tensorflow {
namespace serving {
namespace net_http {

// Creates a server implemented based on the libevents library.
// Returns nullptr if there is any error.
//
// Must call WaitForTermination() or WaitForTerminationWithTimeout() before
// the server is to be destructed.
inline std::unique_ptr<HTTPServerInterface> CreateEvHTTPServer(
    std::unique_ptr<ServerOptions> options) {
  auto server = absl::make_unique<EvHTTPServer>(std::move(options));
  bool result = server->Initialize();
  if (!result) {
    return nullptr;
  }

  return std::move(server);
}

}  // namespace net_http
}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_UTIL_NET_HTTP_SERVER_PUBLIC_HTTPSERVER_H_
