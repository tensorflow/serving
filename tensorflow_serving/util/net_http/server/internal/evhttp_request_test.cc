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

#include "tensorflow_serving/util/net_http/server/internal/evhttp_server.h"

#include <cstdint>
#include <memory>

#include <gtest/gtest.h>

#include "absl/memory/memory.h"
#include "absl/synchronization/internal/thread_pool.h"

#include "tensorflow_serving/util/net_http/client/evhttp_connection.h"
#include "tensorflow_serving/util/net_http/server/public/httpserver.h"
#include "tensorflow_serving/util/net_http/server/public/httpserver_interface.h"
#include "tensorflow_serving/util/net_http/server/public/server_request_interface.h"

namespace tensorflow {
namespace serving {
namespace net_http {
namespace {

class ThreadPool final : public EventExecutor {
 public:
  explicit ThreadPool(int num_threads) : thread_pool_(num_threads) {}

  void Schedule(std::function<void()> fn) override {
    thread_pool_.Schedule(fn);
  }

 private:
  absl::synchronization_internal::ThreadPool thread_pool_;
};

class EvHTTPRequestTest : public ::testing::Test {
 public:
  void SetUp() override { InitServer(); }

  void TearDown() override {
    if (!server->is_terminating()) {
      server->Terminate();
      server->WaitForTermination();
    }
  }

 protected:
  std::unique_ptr<HTTPServerInterface> server;

 private:
  void InitServer() {
    auto options = absl::make_unique<ServerOptions>();
    options->AddPort(0);
    options->SetExecutor(absl::make_unique<ThreadPool>(4));

    server = CreateEvHTTPServer(std::move(options));

    ASSERT_TRUE(server != nullptr);
  }
};

// Test basic GET with 404
TEST_F(EvHTTPRequestTest, SimpleGETNotFound) {
  server->StartAcceptingRequests();

  auto connection =
      EvHTTPConnection::Connect("localhost", server->listen_port());
  ASSERT_TRUE(connection != nullptr);

  ClientRequest request = {"/noop", "GET", {}, nullptr};
  ClientResponse response = {};

  EXPECT_TRUE(connection->BlockingSendRequest(request, &response));
  EXPECT_EQ(response.status, 404);
  EXPECT_FALSE(response.body.empty());

  server->Terminate();
  server->WaitForTermination();
}

// Test basic GET with 200
TEST_F(EvHTTPRequestTest, SimpleGETOK) {
  auto handler = [](ServerRequestInterface* request) {
    request->WriteResponseString("OK");
    request->Reply();
  };
  server->RegisterRequestHandler("/ok", std::move(handler),
                                 RequestHandlerOptions());
  server->StartAcceptingRequests();

  auto connection =
      EvHTTPConnection::Connect("localhost", server->listen_port());
  ASSERT_TRUE(connection != nullptr);

  ClientRequest request = {"/ok", "GET", {}, nullptr};
  ClientResponse response = {};

  EXPECT_TRUE(connection->BlockingSendRequest(request, &response));
  EXPECT_EQ(response.status, 200);
  EXPECT_EQ(response.body, "OK");

  server->Terminate();
  server->WaitForTermination();
}

// Test basic POST with 200
TEST_F(EvHTTPRequestTest, SimplePOST) {
  auto handler = [](ServerRequestInterface* request) {
    int64_t num_bytes;
    auto request_chunk = request->ReadRequestBytes(&num_bytes);
    while (request_chunk != nullptr) {
      request->WriteResponseBytes(request_chunk.get(), num_bytes);
      request_chunk = request->ReadRequestBytes(&num_bytes);
    }
    request->Reply();
  };
  server->RegisterRequestHandler("/ok", std::move(handler),
                                 RequestHandlerOptions());
  server->StartAcceptingRequests();

  auto connection =
      EvHTTPConnection::Connect("localhost", server->listen_port());
  ASSERT_TRUE(connection != nullptr);

  ClientRequest request = {"/ok", "POST", {}, "abcde"};
  ClientResponse response = {};

  EXPECT_TRUE(connection->BlockingSendRequest(request, &response));
  EXPECT_EQ(response.status, 200);
  EXPECT_EQ(response.body, "abcde");

  server->Terminate();
  server->WaitForTermination();
}

// Test request headers
TEST_F(EvHTTPRequestTest, RequestHeaders) {
  auto handler = [](ServerRequestInterface* request) {
    EXPECT_GT(request->request_headers().size(), 2);
    EXPECT_EQ(request->GetRequestHeader("H1"), "v1");
    EXPECT_EQ(request->GetRequestHeader("h1"), "v1");
    EXPECT_EQ(request->GetRequestHeader("H2"), "v2");
    request->WriteResponseString("OK");
    request->Reply();
  };
  server->RegisterRequestHandler("/ok", std::move(handler),
                                 RequestHandlerOptions());
  server->StartAcceptingRequests();

  auto connection =
      EvHTTPConnection::Connect("localhost", server->listen_port());
  ASSERT_TRUE(connection != nullptr);

  ClientRequest request = {"/ok",
                           "GET",
                           {ClientRequest::HeaderKeyValue("H1", "v1"),
                            ClientRequest::HeaderKeyValue("H2", "v2")},
                           nullptr};
  ClientResponse response = {};

  EXPECT_TRUE(connection->BlockingSendRequest(request, &response));
  EXPECT_EQ(response.status, 200);
  EXPECT_EQ(response.body, "OK");

  server->Terminate();
  server->WaitForTermination();
}

// Test response headers
TEST_F(EvHTTPRequestTest, ResponseHeaders) {
  auto handler = [](ServerRequestInterface* request) {
    request->AppendResponseHeader("H1", "V1");
    request->AppendResponseHeader("H2", "V2");
    request->OverwriteResponseHeader("h2", "v2");
    request->WriteResponseString("OK");
    request->Reply();
  };
  server->RegisterRequestHandler("/ok", std::move(handler),
                                 RequestHandlerOptions());
  server->StartAcceptingRequests();

  auto connection =
      EvHTTPConnection::Connect("localhost", server->listen_port());
  ASSERT_TRUE(connection != nullptr);

  ClientRequest request = {"/ok", "GET", {}, nullptr};
  ClientResponse response = {};

  EXPECT_TRUE(connection->BlockingSendRequest(request, &response));
  for (auto keyvalue : response.headers) {
    if (keyvalue.first == "H1") {
      EXPECT_EQ(keyvalue.second, "V1");
    } else if (keyvalue.first == "H2") {
      FAIL() << "H2 should have been overwritten by h2";
    } else if (keyvalue.first == "h2") {
      EXPECT_EQ(keyvalue.second, "v2");
    }
  }
  EXPECT_EQ(response.status, 200);
  EXPECT_EQ(response.body, "OK");

  server->Terminate();
  server->WaitForTermination();
}

}  // namespace
}  // namespace net_http
}  // namespace serving
}  // namespace tensorflow
