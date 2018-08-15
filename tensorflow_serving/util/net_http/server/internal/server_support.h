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

// This pure interface provides callbacks from a request object to the
// server object so the two are properly decoupled.
// This may turn out to be generally useful with no libevents specifics.

#ifndef TENSORFLOW_SERVING_UTIL_NET_HTTP_SERVER_INTERNAL_SERVER_SUPPORT_H_
#define TENSORFLOW_SERVING_UTIL_NET_HTTP_SERVER_INTERNAL_SERVER_SUPPORT_H_

#include <functional>

#include "tensorflow_serving/util/net_http/server/public/server_request_interface.h"

namespace tensorflow {
namespace serving {
namespace net_http {

class ServerSupport {
 public:
  virtual ~ServerSupport() = default;

  ServerSupport(const ServerSupport& other) = delete;
  ServerSupport& operator=(const ServerSupport& other) = delete;

  // book-keeping of active requests
  virtual void IncOps() = 0;
  virtual void DecOps() = 0;

  // Schedules the callback function to run immediately from the event loop.
  // Returns false if any error.
  virtual bool EventLoopSchedule(std::function<void()> fn) = 0;

 protected:
  ServerSupport() = default;
};

}  // namespace net_http
}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_UTIL_NET_HTTP_SERVER_INTERNAL_SERVER_SUPPORT_H_
