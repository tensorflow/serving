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
#include "absl/synchronization/notification.h"
#include "tensorflow_serving/util/net_http/client/test_client/internal/evhttp_connection.h"
#include "tensorflow_serving/util/net_http/internal/fixed_thread_pool.h"
#include "tensorflow_serving/util/net_http/server/public/httpserver.h"
#include "tensorflow_serving/util/net_http/server/public/httpserver_interface.h"
#include "tensorflow_serving/util/net_http/server/public/server_request_interface.h"

namespace tensorflow {
namespace serving {
namespace net_http {
namespace {

class MyExecutor final : public EventExecutor {
 public:
  explicit MyExecutor(int num_threads) : thread_pool_(num_threads) {}

  void Schedule(std::function<void()> fn) override {
    thread_pool_.Schedule(fn);
  }

 private:
  FixedThreadPool thread_pool_;
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
    options->SetExecutor(absl::make_unique<MyExecutor>(4));

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
      TestEvHTTPConnection::Connect("localhost", server->listen_port());
  ASSERT_TRUE(connection != nullptr);

  TestClientRequest request = {"/ok?a=foo", "GET", {}, ""};
  TestClientResponse response = {};

  EXPECT_TRUE(connection->BlockingSendRequest(request, &response));
  EXPECT_EQ(response.status, HTTPStatusCode::OK);
  EXPECT_EQ(response.body, "OK");

  // no canonicalization for the trailing "/"
  request = {"/ok/", "GET", {}, ""};
  response = {};

  EXPECT_TRUE(connection->BlockingSendRequest(request, &response));
  EXPECT_EQ(response.status, HTTPStatusCode::NOT_FOUND);

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
      TestEvHTTPConnection::Connect("localhost", server->listen_port());
  ASSERT_TRUE(connection != nullptr);

  TestClientRequest request = {"/ok", "GET", {}, ""};
  TestClientResponse response = {};

  EXPECT_TRUE(connection->BlockingSendRequest(request, &response));
  EXPECT_EQ(response.status, HTTPStatusCode::OK);
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
      TestEvHTTPConnection::Connect("localhost", server->listen_port());
  ASSERT_TRUE(connection != nullptr);

  TestClientRequest request = {"/ok", "GET", {}, ""};
  TestClientResponse response = {};

  EXPECT_TRUE(connection->BlockingSendRequest(request, &response));
  EXPECT_EQ(response.status, HTTPStatusCode::OK);
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
      TestEvHTTPConnection::Connect("localhost", server->listen_port());
  ASSERT_TRUE(connection != nullptr);

  TestClientRequest request = {"/ok", "GET", {}, ""};
  TestClientResponse response = {};

  EXPECT_TRUE(connection->BlockingSendRequest(request, &response));
  EXPECT_EQ(response.status, HTTPStatusCode::OK);
  EXPECT_EQ(response.body, "OK1");

  request = {"/okxx", "GET", {}, ""};
  response = {};

  EXPECT_TRUE(connection->BlockingSendRequest(request, &response));
  EXPECT_EQ(response.status, HTTPStatusCode::OK);
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
      TestEvHTTPConnection::Connect("localhost", server->listen_port());
  ASSERT_TRUE(connection != nullptr);

  TestClientRequest request = {"/ok", "GET", {}, ""};
  TestClientResponse response = {};

  EXPECT_TRUE(connection->BlockingSendRequest(request, &response));
  EXPECT_EQ(response.status, HTTPStatusCode::OK);
  EXPECT_EQ(response.body, "OK1");

  server->Terminate();
  server->WaitForTermination();
}

// Test handler interaction
TEST_F(EvHTTPServerTest, RequestHandlerInteraction) {
  absl::Notification handler1_start;
  auto handler1 = [&handler1_start](ServerRequestInterface* request) {
    handler1_start.WaitForNotification();
    request->WriteResponseString("OK1");
    request->Reply();
  };

  server->RegisterRequestHandler("/ok", std::move(handler1),
                                 RequestHandlerOptions());

  server->StartAcceptingRequests();

  auto connection =
      TestEvHTTPConnection::Connect("localhost", server->listen_port());
  ASSERT_TRUE(connection != nullptr);
  connection->SetExecutor(absl::make_unique<MyExecutor>(4));

  absl::Notification response_done;
  TestClientRequest request = {"/ok", "GET", {}, ""};
  TestClientResponse response = {};
  response.done = [&response_done]() { response_done.Notify(); };

  EXPECT_TRUE(connection->SendRequest(request, &response));
  EXPECT_FALSE(
      response_done.WaitForNotificationWithTimeout(absl::Milliseconds(50)));

  handler1_start.Notify();
  response_done.WaitForNotification();

  EXPECT_EQ(response.status, HTTPStatusCode::OK);
  EXPECT_EQ(response.body, "OK1");

  connection->Terminate();

  server->Terminate();
  server->WaitForTermination();
}

// Test active-request count during shutdown
TEST_F(EvHTTPServerTest, ActiveRequestCountInShutdown) {
  absl::Notification handler_enter;
  absl::Notification handler_start;
  auto handler = [&handler_enter,
                  &handler_start](ServerRequestInterface* request) {
    handler_enter.Notify();
    handler_start.WaitForNotification();
    request->WriteResponseString("OK1");
    request->Reply();
  };

  server->RegisterRequestHandler("/ok", std::move(handler),
                                 RequestHandlerOptions());

  server->StartAcceptingRequests();

  auto connection =
      TestEvHTTPConnection::Connect("localhost", server->listen_port());
  ASSERT_TRUE(connection != nullptr);
  connection->SetExecutor(absl::make_unique<MyExecutor>(4));

  TestClientRequest request = {"/ok", "GET", {}, ""};
  TestClientResponse response = {};

  EXPECT_TRUE(connection->SendRequest(request, &response));
  handler_enter.WaitForNotification();

  server->Terminate();
  EXPECT_FALSE(server->WaitForTerminationWithTimeout(absl::Milliseconds(50)));

  handler_start.Notify();
  EXPECT_TRUE(server->WaitForTerminationWithTimeout(absl::Milliseconds(5000)));

  connection->Terminate();

  // response.status etc are undefined as the server is terminated
}

}  // namespace
}  // namespace net_http
}  // namespace serving
}  // namespace tensorflow
