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

// libevent based client implementation

#include "tensorflow_serving/util/net_http/client/evhttp_connection.h"

#include "absl/base/internal/raw_logging.h"
#include "absl/strings/str_cat.h"

namespace tensorflow {
namespace serving {
namespace net_http {

EvHTTPConnection::~EvHTTPConnection() {
  if (evcon != nullptr) {
    evhttp_connection_free(evcon);
  }
  if (http_uri != nullptr) {
    evhttp_uri_free(http_uri);
  }

  event_base_free(ev_base);
}

std::unique_ptr<EvHTTPConnection> EvHTTPConnection::Connect(
    absl::string_view url) {
  std::string url_str(url.data(), url.size());
  struct evhttp_uri* http_uri = evhttp_uri_parse(url_str.c_str());
  if (http_uri == nullptr) {
    ABSL_RAW_LOG(ERROR, "Failed to connect : event_base_new()");
    return nullptr;
  }

  const char* host = evhttp_uri_get_host(http_uri);
  if (host == nullptr) {
    ABSL_RAW_LOG(ERROR, "url must have a host %.*s",
                 static_cast<int>(url.size()), url.data());
    return nullptr;
  }

  int port = evhttp_uri_get_port(http_uri);
  if (port == -1) {
    port = 80;
  }

  auto result = Connect(host, port);
  evhttp_uri_free(http_uri);

  return result;
}

std::unique_ptr<EvHTTPConnection> EvHTTPConnection::Connect(
    absl::string_view host, int port) {
  std::unique_ptr<EvHTTPConnection> result(new EvHTTPConnection());

  result->ev_base = event_base_new();
  if (result->ev_base == nullptr) {
    ABSL_RAW_LOG(ERROR, "Failed to connect : event_base_new()");
    return nullptr;
  }

  // blocking call (DNS resolution)
  std::string host_str(host.data(), host.size());
  result->evcon = evhttp_connection_base_bufferevent_new(
      result->ev_base, nullptr, nullptr, host_str.c_str(),
      static_cast<uint16_t>(port));
  if (result->evcon == nullptr) {
    ABSL_RAW_LOG(
        ERROR, "Failed to connect : evhttp_connection_base_bufferevent_new()");
    return nullptr;
  }

  evhttp_connection_set_retries(result->evcon, 0);

  // TODO(wenboz): make this an option (default to 5s)
  evhttp_connection_set_timeout(result->evcon, 5);

  return result;
}

namespace {

void response_cb(struct evhttp_request* req, void* ctx) {
  ClientResponse* response = reinterpret_cast<ClientResponse*>(ctx);

  if (req == nullptr) {
    // TODO(wenboz): make this a util and check safety
    int errcode = EVUTIL_SOCKET_ERROR();
    ABSL_RAW_LOG(ERROR, "socket error = %s (%d)",
                 evutil_socket_error_to_string(errcode), errcode);
    return;
  }

  response->status = evhttp_request_get_response_code(req);

  struct evkeyvalq* headers = evhttp_request_get_input_headers(req);
  struct evkeyval* header;
  for (header = headers->tqh_first; header; header = header->next.tqe_next) {
    response->headers.emplace_back(header->key, header->value);
  }

  char buffer[1024];
  int nread;

  while ((nread = evbuffer_remove(evhttp_request_get_input_buffer(req), buffer,
                                  sizeof(buffer))) > 0) {
    absl::StrAppend(&response->body,
                    absl::string_view(buffer, static_cast<size_t>(nread)));
  }
}

evhttp_cmd_type GetMethodEnum(absl::string_view method, bool with_body) {
  if (method.compare("GET") == 0) {
    return EVHTTP_REQ_GET;
  } else if (method.compare("POST") == 0) {
    return EVHTTP_REQ_POST;
  } else if (method.compare("HEAD") == 0) {
    return EVHTTP_REQ_HEAD;
  } else if (method.compare("PUT") == 0) {
    return EVHTTP_REQ_PUT;
  } else if (method.compare("DELETE") == 0) {
    return EVHTTP_REQ_DELETE;
  } else if (method.compare("OPTIONS") == 0) {
    return EVHTTP_REQ_OPTIONS;
  } else if (method.compare("TRACE") == 0) {
    return EVHTTP_REQ_TRACE;
  } else if (method.compare("CONNECT") == 0) {
    return EVHTTP_REQ_CONNECT;
  } else if (method.compare("PATCH") == 0) {
    return EVHTTP_REQ_PATCH;
  } else {
    if (with_body) {
      return EVHTTP_REQ_POST;
    } else {
      return EVHTTP_REQ_GET;
    }
  }
}

}  // namespace

// Sends the request and has the connection closed
bool EvHTTPConnection::SendRequest(const ClientRequest& request,
                                   ClientResponse* response) {
  struct evhttp_request* req = evhttp_request_new(response_cb, response);
  if (req == nullptr) {
    ABSL_RAW_LOG(ERROR, "Failed to send request : evhttp_request_new()");
    return false;
  }

  struct evkeyvalq* output_headers = evhttp_request_get_output_headers(req);
  for (auto header : request.headers) {
    std::string key(header.first.data(), header.first.size());
    std::string value(header.second.data(), header.second.size());
    evhttp_add_header(output_headers, key.c_str(), value.c_str());
  }

  evhttp_add_header(output_headers, "Connection", "close");

  if (!request.body.empty()) {
    struct evbuffer* output_buffer = evhttp_request_get_output_buffer(req);

    std::string body(request.body.data(), request.body.size());
    evbuffer_add(output_buffer, body.c_str(), request.body.size());

    char length_header[16];
    evutil_snprintf(length_header, sizeof(length_header) - 1, "%lu",
                    request.body.size());
    evhttp_add_header(output_headers, "Content-Length", length_header);
  }

  std::string uri(request.uri_path.data(), request.uri_path.size());
  int r = evhttp_make_request(
      evcon, req, GetMethodEnum(request.method, !request.body.empty()),
      uri.c_str());
  if (r != 0) {
    ABSL_RAW_LOG(ERROR, "evhttp_make_request() failed");
    return false;
  }

  event_base_dispatch(ev_base);

  return true;
}

}  // namespace net_http
}  // namespace serving
}  // namespace tensorflow
