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

// libevent based server implementation

#ifndef TENSORFLOW_SERVING_UTIL_NET_HTTP_SERVER_INTERNAL_EVHTTP_SERVER_H_
#define TENSORFLOW_SERVING_UTIL_NET_HTTP_SERVER_INTERNAL_EVHTTP_SERVER_H_

#include <cstdint>
#include <ctime>
#include <memory>
#include <unordered_map>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"

#include "absl/synchronization/notification.h"

#include "tensorflow_serving/util/net_http/server/internal/evhttp_request.h"
#include "tensorflow_serving/util/net_http/server/internal/server_support.h"
#include "tensorflow_serving/util/net_http/server/public/httpserver_interface.h"

struct event_base;
struct evhttp;
struct evhttp_bound_socket;
struct evhttp_request;

namespace tensorflow {
namespace serving {
namespace net_http {

class EvHTTPServer final : public HTTPServerInterface, ServerSupport {
 public:
  virtual ~EvHTTPServer();

  EvHTTPServer(const EvHTTPServer& other) = delete;
  EvHTTPServer& operator=(const EvHTTPServer& other) = delete;

  explicit EvHTTPServer(std::unique_ptr<ServerOptions> options);

  bool Initialize();

  bool StartAcceptingRequests() override;

  bool is_accepting_requests() const override;

  int listen_port() const override;

  void Terminate() override;

  bool is_terminating() const override;

  void WaitForTermination() override;

  bool WaitForTerminationWithTimeout(absl::Duration timeout) override;

  void RegisterRequestHandler(absl::string_view uri, RequestHandler handler,
                              const RequestHandlerOptions& options) override;

  void RegisterRequestDispatcher(RequestDispatcher dispatcher,
                                 const RequestHandlerOptions& options) override;

  void IncOps() override;
  void DecOps() override;

  bool EventLoopSchedule(std::function<void()> fn) override;

 private:
  static void DispatchEvRequestFn(struct evhttp_request* req, void* server);

  void DispatchEvRequest(struct evhttp_request* req);

  void ScheduleHandlerReference(const RequestHandler& handler,
                                EvHTTPRequest* ev_request)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(request_mu_);
  void ScheduleHandler(RequestHandler&& handler, EvHTTPRequest* ev_request)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(request_mu_);

  struct UriHandlerInfo {
   public:
    UriHandlerInfo(absl::string_view uri_in, RequestHandler handler_in,
                   const RequestHandlerOptions& options_in);
    const std::string uri;
    const RequestHandler handler;
    const RequestHandlerOptions options;
  };

  struct DispatcherInfo {
   public:
    DispatcherInfo(RequestDispatcher dispatcher_in,
                   const RequestHandlerOptions& options_in);

    const RequestDispatcher dispatcher;
    const RequestHandlerOptions options;
  };

  std::unique_ptr<ServerOptions> server_options_;

  // Started accepting requests.
  absl::Notification accepting_requests_;
  // Listener port
  int port_ = 0;

  // Started terminating the server, i.e. Terminate() has been called
  absl::Notification terminating_;

  // Tracks the # of pending operations
  mutable absl::Mutex ops_mu_;
  int64_t num_pending_ops_ ABSL_GUARDED_BY(ops_mu_) = 0;

  mutable absl::Mutex request_mu_;
  std::unordered_map<std::string, UriHandlerInfo> uri_handlers_
      ABSL_GUARDED_BY(request_mu_);
  std::vector<DispatcherInfo> dispatchers_ ABSL_GUARDED_BY(request_mu_);

  // ev instances
  event_base* ev_base_ = nullptr;
  evhttp* ev_http_ = nullptr;
  evhttp_bound_socket* ev_listener_ = nullptr;

  // Timeval used to register immediate callbacks, which are called
  // in the order that they are registered.
  const timeval* immediate_;
};

}  // namespace net_http
}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_UTIL_NET_HTTP_SERVER_INTERNAL_EVHTTP_SERVER_H_
