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
#ifndef TENSORFLOW_SERVING_MODEL_SERVERS_HTTP_SERVER_H_
#define TENSORFLOW_SERVING_MODEL_SERVERS_HTTP_SERVER_H_

#include <memory>

#include "tensorflow_serving/config/monitoring_config.pb.h"
#include "tensorflow_serving/util/net_http/server/public/httpserver_interface.h"

namespace tensorflow {
namespace serving {

class ServerCore;

// Returns a HTTP Server that has following endpoints:
//
//   o HTTP/REST API (under /v1/models/...)
//
// The returned server is in a state of accepting new requests.
std::unique_ptr<net_http::HTTPServerInterface> CreateAndStartHttpServer(
    int port, int num_threads, int timeout_in_ms,
    const MonitoringConfig& monitoring_config, ServerCore* core);

}  // namespace serving
}  // namespace tensorflow
#endif  // TENSORFLOW_SERVING_MODEL_SERVERS_HTTP_SERVER_H_
