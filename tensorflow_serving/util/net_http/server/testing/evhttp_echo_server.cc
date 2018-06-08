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

// A single-threaded server to print the request details as HTML
// URI: /print

#include <cstddef>
#include <cstdint>
#include <cstring>

#include <iostream>
#include <memory>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include "tensorflow_serving/util/net_http/server/public/httpserver.h"
#include "tensorflow_serving/util/net_http/server/public/httpserver_interface.h"
#include "tensorflow_serving/util/net_http/server/public/server_request_interface.h"

namespace {

using absl::StrAppend;

using tensorflow::serving::net_http::EventExecutor;
using tensorflow::serving::net_http::HTTPServerInterface;
using tensorflow::serving::net_http::RequestHandlerOptions;
using tensorflow::serving::net_http::ServerOptions;
using tensorflow::serving::net_http::ServerRequestInterface;
using tensorflow::serving::net_http::SetContentTypeHTML;

void EchoHandler(ServerRequestInterface* req) {
  std::string response;

  StrAppend(&response,
            "<!DOCTYPE html>\n"
            "<html>\n <head>\n"
            "  <meta charset='utf-8'>\n"
            "  <title>Request Echo HTTP Server</title>\n"
            " </head>\n"
            " <body>\n"
            "  <h1>Print the HTTP request detail</h1>\n"
            "  <ul>\n");

  StrAppend(&response, "HTTP Method: ", req->http_method(), " <br>\n");
  StrAppend(&response, "Request Uri: ", req->uri_path(), " <br>\n");

  StrAppend(&response, "<br><br>====<br><br>\n");
  for (auto header : req->request_headers()) {
    StrAppend(&response, header, ": ", req->GetRequestHeader(header), "<br>\n");
  }

  // read the request body
  int64_t num_bytes;
  auto request_chunk = req->ReadRequestBytes(&num_bytes);
  bool print_break = false;
  while (request_chunk != nullptr) {
    if (!print_break) {
      StrAppend(&response, "<br><br>====<br><br>\n");
      print_break = true;
    }
    StrAppend(&response, absl::string_view(request_chunk.get(),
                                           static_cast<size_t>(num_bytes)));
    request_chunk = req->ReadRequestBytes(&num_bytes);
  }

  req->WriteResponseString(response);

  SetContentTypeHTML(req);
  req->Reply();
}

// An executor that runs the current thread, testing only
class MyExcecutor final : public EventExecutor {
 public:
  MyExcecutor() = default;

  void Schedule(std::function<void()> fn) override { fn(); }
};

// Returns the server if success, or nullptr if there is any error.
std::unique_ptr<HTTPServerInterface> StartServer(int port) {
  auto options = absl::make_unique<ServerOptions>();
  options->AddPort(port);
  options->SetExecutor(absl::make_unique<MyExcecutor>());

  auto server = CreateEvHTTPServer(std::move(options));

  if (server == nullptr) {
    return nullptr;
  }

  RequestHandlerOptions handler_options;
  server->RegisterRequestHandler("/print", EchoHandler, handler_options);

  // Blocking here with the use of MyExecutor
  bool success = server->StartAcceptingRequests();

  if (success) {
    return server;
  }

  return nullptr;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: http-server <port:8080>" << std::endl;
    return 1;
  }

  int port = 8080;
  bool port_parsed = absl::SimpleAtoi(argv[1], &port);
  if (!port_parsed) {
    std::cerr << "Invalid port: " << argv[1] << std::endl;
  }

  auto server = StartServer(port);

  if (server != nullptr) {
    server->WaitForTermination();
    return 0;
  } else {
    std::cerr << "Failed to start the server." << std::endl;
    return 1;
  }
}
