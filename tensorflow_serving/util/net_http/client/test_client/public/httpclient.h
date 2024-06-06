/* Copyright 2020 Google Inc. All Rights Reserved.

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

#ifndef THIRD_PARTY_TENSORFLOW_SERVING_UTIL_NET_HTTP_CLIENT_TEST_CLIENT_PUBLIC_HTTPCLIENT_H_
#define THIRD_PARTY_TENSORFLOW_SERVING_UTIL_NET_HTTP_CLIENT_TEST_CLIENT_PUBLIC_HTTPCLIENT_H_

#include "absl/memory/memory.h"
#include "tensorflow_serving/util/net_http/client/test_client/internal/evhttp_connection.h"
#include "tensorflow_serving/util/net_http/client/test_client/public/httpclient_interface.h"

// Factory to manage internal dependency
// NOTE: This API is not yet finalized, and should in its current state be
// considered experimental

namespace tensorflow {
namespace serving {
namespace net_http {

// Creates a connection to a server implemented based on the libevents library.
// Returns nullptr if there is any error.
inline std::unique_ptr<TestHTTPClientInterface> CreateEvHTTPConnection(
    absl::string_view host, int port) {
  auto connection = absl::make_unique<TestEvHTTPConnection>();
  connection = connection->Connect(host, port);
  if (!connection) {
    return nullptr;
  }

  return std::move(connection);
}

}  // namespace net_http
}  // namespace serving
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_SERVING_UTIL_NET_HTTP_CLIENT_TEST_CLIENT_PUBLIC_HTTPCLIENT_H_

