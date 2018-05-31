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

// API for the HTTP client

#ifndef TENSORFLOW_SERVING_UTIL_NET_HTTP_CLIENT_EVHTTP_CONNECTION_H_
#define TENSORFLOW_SERVING_UTIL_NET_HTTP_CLIENT_EVHTTP_CONNECTION_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"

#include "libevent/include/event2/buffer.h"
#include "libevent/include/event2/bufferevent.h"
#include "libevent/include/event2/event.h"
#include "libevent/include/event2/http.h"
#include "libevent/include/event2/keyvalq_struct.h"
#include "libevent/include/event2/util.h"

namespace tensorflow {
namespace serving {
namespace net_http {

// The following types may be moved to an API interface in future.

// Data to be copied
struct ClientRequest {
  typedef std::pair<absl::string_view, absl::string_view> HeaderKeyValue;

  absl::string_view uri_path;
  absl::string_view method;  // must be in upper-case
  std::vector<HeaderKeyValue> headers;
  absl::string_view body;
};

// Caller allocates the data for output
struct ClientResponse {
  typedef std::pair<std::string, std::string> HeaderKeyValue;

  int status = 0;
  std::vector<HeaderKeyValue> headers;
  std::string body;
};

class EvHTTPConnection final {
 public:
  ~EvHTTPConnection();

  EvHTTPConnection(const EvHTTPConnection& other) = delete;
  EvHTTPConnection& operator=(const EvHTTPConnection& other) = delete;

  // Returns a new connection given an absolute URL.
  // Always treat the URL scheme as "http" for now.
  // Returns nullptr if any error
  static std::unique_ptr<EvHTTPConnection> Connect(absl::string_view url);

  // Returns a new connection to the specified host:port.
  // Returns nullptr if any error
  static std::unique_ptr<EvHTTPConnection> Connect(absl::string_view host,
                                                   int port);

  // Returns a new connection to the specified port of localhost.
  // Returns nullptr if any error
  static std::unique_ptr<EvHTTPConnection> ConnectLocal(int port) {
    return Connect("localhost", port);
  }

  // Sends a request and blocks the caller till a response is received
  // or any error has happened.
  // Returns false if any error.
  bool SendRequest(const ClientRequest& request, ClientResponse* response);

 private:
  EvHTTPConnection() = default;

  struct event_base* ev_base;
  struct evhttp_uri* http_uri;
  struct evhttp_connection* evcon;
};

}  // namespace net_http
}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_UTIL_NET_HTTP_CLIENT_EVHTTP_CONNECTION_H_
