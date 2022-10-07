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

#include <cstdint>
#include <memory>
#include <random>

#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "tensorflow_serving/util/net_http/client/test_client/internal/evhttp_connection.h"
#include "tensorflow_serving/util/net_http/compression/gzip_zlib.h"
#include "tensorflow_serving/util/net_http/internal/fixed_thread_pool.h"
#include "tensorflow_serving/util/net_http/public/response_code_enum.h"
#include "tensorflow_serving/util/net_http/server/internal/evhttp_server.h"
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
    options->SetExecutor(absl::make_unique<MyExecutor>(4));

    server = CreateEvHTTPServer(std::move(options));

    ASSERT_TRUE(server != nullptr);
  }
};

// Test basic GET with 404
TEST_F(EvHTTPRequestTest, SimpleGETNotFound) {
  server->StartAcceptingRequests();

  auto connection =
      TestEvHTTPConnection::Connect("localhost", server->listen_port());
  ASSERT_TRUE(connection != nullptr);

  TestClientRequest request = {"/noop", "GET", {}, ""};
  TestClientResponse response = {};

  EXPECT_TRUE(connection->BlockingSendRequest(request, &response));
  EXPECT_EQ(response.status, HTTPStatusCode::NOT_FOUND);
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
      TestEvHTTPConnection::Connect("localhost", server->listen_port());
  ASSERT_TRUE(connection != nullptr);

  TestClientRequest request = {"/ok", "POST", {}, "abcde"};
  TestClientResponse response = {};

  EXPECT_TRUE(connection->BlockingSendRequest(request, &response));
  EXPECT_EQ(response.status, HTTPStatusCode::OK);
  EXPECT_EQ(response.body, "abcde");

  server->Terminate();
  server->WaitForTermination();
}

// Test request's uri_path() method.
TEST_F(EvHTTPRequestTest, RequestUri) {
  static const char* const kUriPath[] = {
      "/",
      "/path",
      "/path/",
      "/path?query=value",
      "/path#fragment",
      "/path?param=value#fragment",
      "/path?param=value%20value",
  };

  int counter = 0;
  auto handler = [&counter](ServerRequestInterface* request) {
    EXPECT_EQ(kUriPath[counter++], request->uri_path());
    request->Reply();
  };
  server->RegisterRequestDispatcher(
      [&handler](ServerRequestInterface* request) -> RequestHandler {
        return handler;
      },
      RequestHandlerOptions());

  server->StartAcceptingRequests();

  auto connection =
      TestEvHTTPConnection::Connect("localhost", server->listen_port());
  ASSERT_TRUE(connection != nullptr);

  for (const char* path : kUriPath) {
    TestClientRequest request = {path, "GET", {}, ""};
    TestClientResponse response = {};

    EXPECT_TRUE(connection->BlockingSendRequest(request, &response));
  }

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
      TestEvHTTPConnection::Connect("localhost", server->listen_port());
  ASSERT_TRUE(connection != nullptr);

  TestClientRequest request = {"/ok",
                               "GET",
                               {TestClientRequest::HeaderKeyValue("H1", "v1"),
                                TestClientRequest::HeaderKeyValue("H2", "v2")},
                               ""};
  TestClientResponse response = {};

  EXPECT_TRUE(connection->BlockingSendRequest(request, &response));
  EXPECT_EQ(response.status, HTTPStatusCode::OK);
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
      TestEvHTTPConnection::Connect("localhost", server->listen_port());
  ASSERT_TRUE(connection != nullptr);

  TestClientRequest request = {"/ok", "GET", {}, ""};
  TestClientResponse response = {};

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
  EXPECT_EQ(response.status, HTTPStatusCode::OK);
  EXPECT_EQ(response.body, "OK");

  server->Terminate();
  server->WaitForTermination();
}

// === gzip support ====

// Test invalid gzip body
TEST_F(EvHTTPRequestTest, InvalidGzipPost) {
  auto handler = [](ServerRequestInterface* request) {
    int64_t num_bytes;
    auto request_body = request->ReadRequestBytes(&num_bytes);
    EXPECT_TRUE(request_body == nullptr);
    EXPECT_EQ(0, num_bytes);

    request->Reply();
  };
  server->RegisterRequestHandler("/ok", std::move(handler),
                                 RequestHandlerOptions());
  server->StartAcceptingRequests();

  auto connection =
      TestEvHTTPConnection::Connect("localhost", server->listen_port());
  ASSERT_TRUE(connection != nullptr);

  TestClientRequest request = {"/ok", "POST", {}, "abcde"};
  request.headers.emplace_back("Content-Encoding", "my_gzip");
  TestClientResponse response = {};

  EXPECT_TRUE(connection->BlockingSendRequest(request, &response));
  EXPECT_EQ(response.status, HTTPStatusCode::OK);

  server->Terminate();
  server->WaitForTermination();
}

// Test disabled gzip
TEST_F(EvHTTPRequestTest, DisableGzipPost) {
  auto handler = [](ServerRequestInterface* request) {
    int64_t num_bytes;
    auto request_body = request->ReadRequestBytes(&num_bytes);
    EXPECT_EQ(5, num_bytes);

    request->Reply();
  };
  RequestHandlerOptions options;
  options.set_auto_uncompress_input(false);
  server->RegisterRequestHandler("/ok", std::move(handler), options);
  server->StartAcceptingRequests();

  auto connection =
      TestEvHTTPConnection::Connect("localhost", server->listen_port());
  ASSERT_TRUE(connection != nullptr);

  TestClientRequest request = {"/ok", "POST", {}, "abcde"};
  request.headers.emplace_back("Content-Encoding", "my_gzip");
  TestClientResponse response = {};

  EXPECT_TRUE(connection->BlockingSendRequest(request, &response));
  EXPECT_EQ(response.status, HTTPStatusCode::OK);

  server->Terminate();
  server->WaitForTermination();
}

std::string CompressLargeString(const char* data, size_t size,
                                size_t buf_size) {
  ZLib zlib;
  std::string buf(buf_size, '\0');
  size_t compressed_size = buf.size();
  zlib.Compress((Bytef*)buf.data(), &compressed_size, (Bytef*)data, size);

  return std::string(buf.data(), compressed_size);
}

std::string CompressString(const char* data, size_t size) {
  return CompressLargeString(data, size, 1024);
}

// Test valid gzip body
TEST_F(EvHTTPRequestTest, ValidGzipPost) {
  constexpr char kBody[] = "abcdefg12345";
  std::string compressed = CompressString(kBody, sizeof(kBody) - 1);

  auto handler = [&](ServerRequestInterface* request) {
    int64_t num_bytes;
    auto request_body = request->ReadRequestBytes(&num_bytes);

    std::string body_str(request_body.get(), static_cast<size_t>(num_bytes));
    EXPECT_EQ(body_str, std::string(kBody));
    EXPECT_EQ(sizeof(kBody) - 1, num_bytes);

    EXPECT_EQ(nullptr, request->ReadRequestBytes(&num_bytes));
    EXPECT_EQ(0, num_bytes);

    request->Reply();
  };
  server->RegisterRequestHandler("/ok", std::move(handler),
                                 RequestHandlerOptions());
  server->StartAcceptingRequests();

  auto connection =
      TestEvHTTPConnection::Connect("localhost", server->listen_port());
  ASSERT_TRUE(connection != nullptr);

  TestClientRequest request = {"/ok", "POST", {}, compressed};
  request.headers.emplace_back("Content-Encoding", "my_gzip");
  TestClientResponse response = {};

  EXPECT_TRUE(connection->BlockingSendRequest(request, &response));
  EXPECT_EQ(response.status, HTTPStatusCode::OK);

  server->Terminate();
  server->WaitForTermination();
}

// Test gzip exceeding the max uncompressed limit
TEST_F(EvHTTPRequestTest, GzipExceedingLimit) {
  constexpr char kBody[] = "abcdefg12345";
  constexpr int bodySize = sizeof(kBody) - 1;
  std::string compressed = CompressString(kBody, static_cast<size_t>(bodySize));

  auto handler = [&](ServerRequestInterface* request) {
    int64_t num_bytes;
    auto request_body = request->ReadRequestBytes(&num_bytes);

    std::string body_str(request_body.get(), static_cast<size_t>(num_bytes));
    EXPECT_TRUE(request_body == nullptr);
    EXPECT_EQ(0, num_bytes);

    request->Reply();
  };

  RequestHandlerOptions options;
  options.set_auto_uncompress_max_size(bodySize - 1);  // not enough buffer
  server->RegisterRequestHandler("/ok", std::move(handler), options);
  server->StartAcceptingRequests();

  auto connection =
      TestEvHTTPConnection::Connect("localhost", server->listen_port());
  ASSERT_TRUE(connection != nullptr);

  TestClientRequest request = {"/ok", "POST", {}, compressed};
  request.headers.emplace_back("Content-Encoding", "my_gzip");
  TestClientResponse response = {};

  EXPECT_TRUE(connection->BlockingSendRequest(request, &response));
  EXPECT_EQ(response.status, HTTPStatusCode::OK);

  server->Terminate();
  server->WaitForTermination();
}

std::string MakeRandomString(int64_t len) {
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<> dis('a', 'z');
  std::string s(len, '0');
  for (char& c : s) {
    c = dis(gen);
  }
  return s;
}

// Test large gzip body
TEST_F(EvHTTPRequestTest, LargeGzipPost) {
  constexpr int64_t uncompress_len = 1024 * 1024;
  std::string uncompressed = MakeRandomString(uncompress_len);
  std::string compressed = CompressLargeString(
      uncompressed.data(), uncompressed.size(), 2 * uncompress_len);

  auto handler = [&](ServerRequestInterface* request) {
    int64_t num_bytes;
    auto request_body = request->ReadRequestBytes(&num_bytes);

    std::string body_str(request_body.get(), static_cast<size_t>(num_bytes));
    EXPECT_EQ(body_str, uncompressed);
    EXPECT_EQ(uncompressed.size(), num_bytes);

    EXPECT_EQ(nullptr, request->ReadRequestBytes(&num_bytes));
    EXPECT_EQ(0, num_bytes);

    request->Reply();
  };
  server->RegisterRequestHandler("/ok", std::move(handler),
                                 RequestHandlerOptions());
  server->StartAcceptingRequests();

  auto connection =
      TestEvHTTPConnection::Connect("localhost", server->listen_port());
  ASSERT_TRUE(connection != nullptr);

  TestClientRequest request = {"/ok", "POST", {}, compressed};
  request.headers.emplace_back("Content-Encoding", "my_gzip");
  TestClientResponse response = {};

  EXPECT_TRUE(connection->BlockingSendRequest(request, &response));
  EXPECT_EQ(response.status, HTTPStatusCode::OK);

  server->Terminate();
  server->WaitForTermination();
}

}  // namespace
}  // namespace net_http
}  // namespace serving
}  // namespace tensorflow
