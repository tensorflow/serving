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

#ifndef THIRD_PARTY_TENSORFLOW_SERVING_UTIL_NET_HTTP_CLIENT_TEST_CLIENT_PUBLIC_HTTPCLIENT_INTERFACE_H_
#define THIRD_PARTY_TENSORFLOW_SERVING_UTIL_NET_HTTP_CLIENT_TEST_CLIENT_PUBLIC_HTTPCLIENT_INTERFACE_H_

#include "tensorflow_serving/util/net_http/public/response_code_enum.h"
#include "tensorflow_serving/util/net_http/server/public/httpserver_interface.h"

// API for the HTTP Client
// NOTE: This API is not yet finalized, and should be considered experimental.

namespace tensorflow {
namespace serving {
namespace net_http {

// Data to be copied
struct TestClientRequest {
  typedef std::pair<absl::string_view, absl::string_view> HeaderKeyValue;

  absl::string_view uri_path;
  absl::string_view method;  // must be in upper-case
  std::vector<HeaderKeyValue> headers;
  absl::string_view body;
};

// Caller allocates the data for output
struct TestClientResponse {
  typedef std::pair<std::string, std::string> HeaderKeyValue;

  HTTPStatusCode status = HTTPStatusCode::UNDEFINED;
  std::vector<HeaderKeyValue> headers;
  std::string body;

  std::function<void()> done;  // callback
};

// This interface class specifies the API contract for the HTTP client.
class TestHTTPClientInterface {
 public:
  TestHTTPClientInterface(const TestHTTPClientInterface& other) = delete;
  TestHTTPClientInterface& operator=(const TestHTTPClientInterface& other) =
      delete;

  virtual ~TestHTTPClientInterface() = default;

  // Terminates the connection.
  virtual void Terminate() = 0;

  // Sends a request and blocks the caller till a response is received
  // or any error has happened.
  // Returns false if any error.
  virtual bool BlockingSendRequest(const TestClientRequest& request,
                                   TestClientResponse* response) = 0;

  // Sends a request and returns immediately. The response will be handled
  // asynchronously via the response->done callback.
  // Returns false if any error in sending the request, or if the executor
  // has not been configured.
  virtual bool SendRequest(const TestClientRequest& request,
                           TestClientResponse* response) = 0;

  // Sets the executor for processing requests asynchronously.
  virtual void SetExecutor(std::unique_ptr<EventExecutor> executor) = 0;

 protected:
  TestHTTPClientInterface() = default;
};

}  // namespace net_http
}  // namespace serving
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_SERVING_UTIL_NET_HTTP_CLIENT_TEST_CLIENT_PUBLIC_HTTPCLIENT_INTERFACE_H_
