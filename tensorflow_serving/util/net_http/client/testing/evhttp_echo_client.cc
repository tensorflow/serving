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

// A test client to print the response from the evhttp_echo_server
// URI: /print

#include <iostream>

#include "tensorflow_serving/util/net_http/client/evhttp_connection.h"

namespace {

using tensorflow::serving::net_http::ClientRequest;
using tensorflow::serving::net_http::ClientResponse;
using tensorflow::serving::net_http::EvHTTPConnection;

bool SendRequest(const char* url) {
  auto connection = EvHTTPConnection::Connect(url);
  if (connection == nullptr) {
    std::cerr << "Fail to connect to %s" << url;
  }

  ClientRequest request = {url, "GET", {}, nullptr};
  ClientResponse response = {};

  if (!connection->BlockingSendRequest(request, &response)) {
    std::cerr << "Request failed.";
    return false;
  }

  std::cout << "Response received: " << std::endl
            << "Status: " << response.status << std::endl;

  for (auto keyval : response.headers) {
    std::cout << keyval.first << " : " << keyval.second << std::endl;
  }

  std::cout << std::endl << response.body << std::endl;
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: http-client <url>" << std::endl;
    return 1;
  }

  return SendRequest(argv[1]);
}
