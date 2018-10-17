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

// APIs for the HTTP server.

#ifndef TENSORFLOW_SERVING_UTIL_NET_HTTP_SERVER_PUBLIC_HTTPSERVER_INTERFACE_H_
#define TENSORFLOW_SERVING_UTIL_NET_HTTP_SERVER_PUBLIC_HTTPSERVER_INTERFACE_H_

#include <cassert>

#include <functional>
#include <memory>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/time/time.h"

#include "tensorflow_serving/util/net_http/server/public/server_request_interface.h"

namespace tensorflow {
namespace serving {
namespace net_http {

// A non-blocking executor for processing I/O polling or callback events.
class EventExecutor {
 public:
  virtual ~EventExecutor() = default;

  EventExecutor(const EventExecutor& other) = delete;
  EventExecutor& operator=(const EventExecutor& other) = delete;

  // Schedule the specified 'fn' for execution in this executor.
  // Must be non-blocking
  virtual void Schedule(std::function<void()> fn) = 0;

 protected:
  EventExecutor() = default;
};

// Options to specify when a server instance is created.
class ServerOptions {
 public:
  ServerOptions() = default;

  // At least one port has to be configured.
  // Port 0 will start the server using an ephemeral port.
  void AddPort(int port) {
    assert(port >= 0);
    ports_.emplace_back(port);
  }

  // The default executor for running I/O event polling.
  // This is a mandatory option.
  void SetExecutor(std::unique_ptr<EventExecutor> executor) {
    executor_ = std::move(executor);
  }

  const std::vector<int>& ports() const { return ports_; }

  EventExecutor* executor() const { return executor_.get(); }

 private:
  std::vector<int> ports_;
  std::unique_ptr<EventExecutor> executor_;
};

// Options to specify when registering a handler (given a uri pattern).
// This should be a value type.
class RequestHandlerOptions {
 public:
  RequestHandlerOptions() = default;

  RequestHandlerOptions(const RequestHandlerOptions&) = default;
  RequestHandlerOptions& operator=(const RequestHandlerOptions&) = default;

  // Sets the max length of uncompressed data when uncompressing a request body
  inline RequestHandlerOptions& set_auto_uncompress_max_size(int64_t size) {
    auto_uncompress_max_size_ = size;
    return *this;
  }

  // The max length of uncompressed data when doing uncompress. Returns 0 if
  // not set. See Zlib::kMaxUncompressedBytes for the default config.
  inline int64_t auto_uncompress_max_size() const {
    return auto_uncompress_max_size_;
  }

  // The auto_uncompress_input option specifies whether the request
  // input data should be uncompressed if the request has the
  // Content-Encoding: .*gzip.* header. The option defaults to true.
  inline RequestHandlerOptions& set_auto_uncompress_input(
      bool should_uncompress) {
    auto_uncompress_input_ = should_uncompress;
    return *this;
  }

  inline bool auto_uncompress_input() const { return auto_uncompress_input_; }

 private:
  // To be added: compression, CORS rules, streaming control
  // thread executor, admission control, limits ...

  bool auto_uncompress_input_ = true;

  int64_t auto_uncompress_max_size_ = 0;
};

// A request handler is registered by the application to handle a request
// based on the request Uri path, available via ServerRequestInterface.
//
// Request handlers need be completely non-blocking. And handlers may add
// callbacks to a thread-pool that is managed by the application itself.
typedef std::function<void(ServerRequestInterface*)> RequestHandler;

// Returns a nullptr if the request is not handled by this dispatcher.
typedef std::function<RequestHandler(ServerRequestInterface*)>
    RequestDispatcher;

// This interface class specifies the API contract for the HTTP server.
//
// Requirements for implementations:
// - must be thread-safe
// - multiple HTTP server instances need be supported in a single process
// - for a basic implementation, the application needs provide a dedicated
//   thread to handle all I/O events, i.e. the thread that calls
//   StartAcceptingRequests().
// - the arrival order of concurrent requests is insignificant because the
//   server runtime and I/O are completely event-driven.
class HTTPServerInterface {
 public:
  virtual ~HTTPServerInterface() = default;

  HTTPServerInterface(const HTTPServerInterface& other) = delete;
  HTTPServerInterface& operator=(const HTTPServerInterface& other) = delete;

  // Starts to accept requests arrived on the network.
  // Returns false if the server runtime fails to initialize properly.
  virtual bool StartAcceptingRequests() = 0;

  // Returns true if StartAcceptingRequests() has been called.
  virtual bool is_accepting_requests() const = 0;

  // Returns the server listener port if any, or else returns 0.
  virtual int listen_port() const = 0;

  // Starts the server termination, and returns immediately.
  virtual void Terminate() = 0;

  // Returns true if Terminate() has been called.
  virtual bool is_terminating() const = 0;

  // Blocks the calling thread until the server is terminated and safe
  // to destroy.
  virtual void WaitForTermination() = 0;

  // Blocks the calling thread until the server is terminated and safe
  // to destroy, or until the specified timeout elapses.  Returns true
  // if safe termination completed within the timeout, and false otherwise.
  virtual bool WaitForTerminationWithTimeout(absl::Duration timeout) = 0;

  // To be specified: lameduck

  // Registers a request handler with exact URI path matching.
  // Any existing handler under the same uri will be overwritten.
  // The server owns the handler after registration.
  // Handlers may be registered after the server has been started.
  virtual void RegisterRequestHandler(absl::string_view uri,
                                      RequestHandler handler,
                                      const RequestHandlerOptions& options) = 0;

  // Registers a request dispatcher, i.e. application-provided URI dispatching
  // logic, e.g. a regexp based one.
  //
  // For a given request, dispatchers are only invoked if there is no exact URI
  // path matching to any registered request handler.
  //
  // Dispatchers are invoked in order of registration, i.e. first registered
  // gets first pick. The server owns the dispatcher after registration.
  // Dispatchers may be registered after the server has been started.
  virtual void RegisterRequestDispatcher(
      RequestDispatcher dispatcher, const RequestHandlerOptions& options) = 0;

  // To be added: unregister (if needed)

 protected:
  HTTPServerInterface() = default;
};

}  // namespace net_http
}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_UTIL_NET_HTTP_SERVER_PUBLIC_HTTPSERVER_INTERFACE_H_
