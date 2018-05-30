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

#include <memory>

#include <gtest/gtest.h>

#include "absl/memory/memory.h"
#include "absl/synchronization/internal/thread_pool.h"

#include "tensorflow_serving/util/net_http/client/evhttp_connection.h"
#include "tensorflow_serving/util/net_http/server/public/httpserver.h"
#include "tensorflow_serving/util/net_http/server/public/httpserverinterface.h"
#include "tensorflow_serving/util/net_http/server/public/serverrequestinterface.h"

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

class EvHTTPServerTest : public ::testing::Test {
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

// Test StartAcceptingRequests
TEST_F(EvHTTPServerTest, AcceptingTerminating) {
  EXPECT_FALSE(server->is_accepting_requests());
  server->StartAcceptingRequests();
  EXPECT_TRUE(server->is_accepting_requests());

  server->Terminate();
  EXPECT_TRUE(server->is_terminating());

  server->WaitForTermination();
}

// Test the path matching behavior
TEST_F(EvHTTPServerTest, ExactPathMatching) {
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

  ClientRequest request = {"/ok?a=foo", "GET", {}, nullptr};
  ClientResponse response;

  EXPECT_TRUE(connection->SendRequest(request, &response));
  EXPECT_EQ(response.status, 200);
  EXPECT_EQ(response.body, "OK");

  // no canonicalization for the trailing "/"
  request = {"/ok/", "GET", {}, nullptr};
  response = {};

  EXPECT_TRUE(connection->SendRequest(request, &response));
  EXPECT_EQ(response.status, 404);

  server->Terminate();
  server->WaitForTermination();
}

// Test RequestHandler overwriting
TEST_F(EvHTTPServerTest, RequestHandlerOverwriting) {
  auto handler1 = [](ServerRequestInterface* request) {
    request->WriteResponseString("OK1");
    request->Reply();
  };
  auto handler2 = [](ServerRequestInterface* request) {
    request->WriteResponseString("OK2");
    request->Reply();
  };

  server->RegisterRequestHandler("/ok", std::move(handler1),
                                 RequestHandlerOptions());
  server->RegisterRequestHandler("/ok", std::move(handler2),
                                 RequestHandlerOptions());
  server->StartAcceptingRequests();

  auto connection =
      EvHTTPConnection::Connect("localhost", server->listen_port());
  ASSERT_TRUE(connection != nullptr);

  ClientRequest request = {"/ok", "GET", {}, nullptr};
  ClientResponse response;

  EXPECT_TRUE(connection->SendRequest(request, &response));
  EXPECT_EQ(response.status, 200);
  EXPECT_EQ(response.body, "OK2");

  server->Terminate();
  server->WaitForTermination();
}

// Test single RequestDispatcher
TEST_F(EvHTTPServerTest, SingleRequestDispather) {
  auto handler = [](ServerRequestInterface* request) {
    request->WriteResponseString("OK");
    request->Reply();
  };

  auto dispatcher = [&handler](ServerRequestInterface* request) {
    return handler;
  };

  server->RegisterRequestDispatcher(std::move(dispatcher),
                                    RequestHandlerOptions());
  server->StartAcceptingRequests();

  auto connection =
      EvHTTPConnection::Connect("localhost", server->listen_port());
  ASSERT_TRUE(connection != nullptr);

  ClientRequest request = {"/ok", "GET", {}, nullptr};
  ClientResponse response;

  EXPECT_TRUE(connection->SendRequest(request, &response));
  EXPECT_EQ(response.status, 200);
  EXPECT_EQ(response.body, "OK");

  server->Terminate();
  server->WaitForTermination();
}

// Test URI path precedes over RequestDispatcher
TEST_F(EvHTTPServerTest, UriPrecedesOverRequestDispather) {
  auto handler1 = [](ServerRequestInterface* request) {
    request->WriteResponseString("OK1");
    request->Reply();
  };

  server->RegisterRequestHandler("/ok", std::move(handler1),
                                 RequestHandlerOptions());

  auto handler2 = [](ServerRequestInterface* request) {
    request->WriteResponseString("OK2");
    request->Reply();
  };

  auto dispatcher = [&handler2](ServerRequestInterface* request) {
    return handler2;
  };
  server->RegisterRequestDispatcher(std::move(dispatcher),
                                    RequestHandlerOptions());

  server->StartAcceptingRequests();

  auto connection =
      EvHTTPConnection::Connect("localhost", server->listen_port());
  ASSERT_TRUE(connection != nullptr);

  ClientRequest request = {"/ok", "GET", {}, nullptr};
  ClientResponse response;

  EXPECT_TRUE(connection->SendRequest(request, &response));
  EXPECT_EQ(response.status, 200);
  EXPECT_EQ(response.body, "OK1");

  request = {"/okxx", "GET", {}, nullptr};
  response = {};

  EXPECT_TRUE(connection->SendRequest(request, &response));
  EXPECT_EQ(response.status, 200);
  EXPECT_EQ(response.body, "OK2");

  server->Terminate();
  server->WaitForTermination();
}

// Test RequestDispatcher in-order dispatching
TEST_F(EvHTTPServerTest, InOrderRequestDispather) {
  auto dispatcher1 = [](ServerRequestInterface* request) {
    return [](ServerRequestInterface* request) {
      request->WriteResponseString("OK1");
      request->Reply();
    };
  };

  auto dispatcher2 = [](ServerRequestInterface* request) {
    return [](ServerRequestInterface* request) {
      request->WriteResponseString("OK2");
      request->Reply();
    };
  };

  server->RegisterRequestDispatcher(std::move(dispatcher1),
                                    RequestHandlerOptions());
  server->RegisterRequestDispatcher(std::move(dispatcher2),
                                    RequestHandlerOptions());

  server->StartAcceptingRequests();

  auto connection =
      EvHTTPConnection::Connect("localhost", server->listen_port());
  ASSERT_TRUE(connection != nullptr);

  ClientRequest request = {"/ok", "GET", {}, nullptr};
  ClientResponse response;

  EXPECT_TRUE(connection->SendRequest(request, &response));
  EXPECT_EQ(response.status, 200);
  EXPECT_EQ(response.body, "OK1");

  server->Terminate();
  server->WaitForTermination();
}

// TODO(wenboz): More tests on clean shutdown

}  // namespace
}  // namespace net_http
}  // namespace serving
}  // namespace tensorflow
