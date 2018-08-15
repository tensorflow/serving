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

#include <dirent.h>
#include <sys/socket.h>
#include <sys/types.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "absl/strings/numbers.h"

#include "libevent/include/event2/buffer.h"
#include "libevent/include/event2/event.h"
#include "libevent/include/event2/http.h"
#include "libevent/include/event2/keyvalq_struct.h"
#include "libevent/include/event2/util.h"

namespace {

char uri_root[512];

void request_cb(struct evhttp_request *req, void *arg) {
  const char *method;

  switch (evhttp_request_get_command(req)) {
    case EVHTTP_REQ_GET:
      method = "GET";
      break;
    case EVHTTP_REQ_POST:
      method = "POST";
      break;
    case EVHTTP_REQ_HEAD:
      method = "HEAD";
      break;
    case EVHTTP_REQ_PUT:
      method = "PUT";
      break;
    case EVHTTP_REQ_DELETE:
      method = "DELETE";
      break;
    case EVHTTP_REQ_OPTIONS:
      method = "OPTIONS";
      break;
    case EVHTTP_REQ_TRACE:
      method = "TRACE";
      break;
    case EVHTTP_REQ_CONNECT:
      method = "CONNECT";
      break;
    case EVHTTP_REQ_PATCH:
      method = "PATCH";
      break;
    default:
      method = "unknown";
      break;
  }

  struct evbuffer *response_body = evbuffer_new();
  if (response_body == nullptr) {
    evhttp_send_error(req, HTTP_SERVUNAVAIL, nullptr);
    return;
  }

  evbuffer_add_printf(response_body,
                      "<!DOCTYPE html>\n"
                      "<html>\n <head>\n"
                      "  <meta charset='utf-8'>\n"
                      "  <title>Mini libevent httpserver</title>\n"
                      " </head>\n"
                      " <body>\n"
                      "  <h1>Print the HTTP request detail</h1>\n"
                      "  <ul>\n");

  evbuffer_add_printf(response_body, "HTTP Method: %s <br>\n", method);

  const char *uri = evhttp_request_get_uri(req);  // no malloc
  evbuffer_add_printf(response_body, "Request Uri: %s <br>\n", uri);

  struct evhttp_uri *decoded_url = evhttp_uri_parse(uri);
  if (decoded_url == nullptr) {
    evhttp_send_error(req, HTTP_BADREQUEST, nullptr);
    return;
  }

  evbuffer_add_printf(response_body, "Decoded Uri:<br>\n");

  evbuffer_add_printf(response_body, "&nbsp;&nbsp;scheme : %s <br>\n",
                      evhttp_uri_get_scheme(decoded_url));
  evbuffer_add_printf(response_body, "&nbsp;&nbsp;host : %s <br>\n",
                      evhttp_uri_get_host(decoded_url));
  evbuffer_add_printf(response_body, "&nbsp;&nbsp;port : %d <br>\n",
                      evhttp_uri_get_port(decoded_url));
  evbuffer_add_printf(response_body, "&nbsp;&nbsp;path : %s <br>\n",
                      evhttp_uri_get_path(decoded_url));
  evbuffer_add_printf(response_body, "&nbsp;&nbsp;query : %s <br>\n",
                      evhttp_uri_get_query(decoded_url));

  const char *path = evhttp_uri_get_path(decoded_url);
  if (path == nullptr) {
    path = "/";
  }

  evbuffer_add_printf(response_body, "Uri path: %s <br>\n", path);

  char *decoded_path = evhttp_uridecode(path, 1, nullptr);
  if (decoded_path == nullptr) {
    evhttp_send_error(req, HTTP_BADREQUEST, nullptr);
    return;
  }

  evbuffer_add_printf(response_body, "Decoded path: %s <br>\n", decoded_path);

  evbuffer_add_printf(response_body, "<br><br>====<br><br>\n");

  struct evkeyvalq *headers = evhttp_request_get_input_headers(req);

  struct evkeyval *header;
  for (header = headers->tqh_first; header; header = header->next.tqe_next) {
    evbuffer_add_printf(response_body, "%s : %s<br>\n", header->key,
                        header->value);
  }

  struct evbuffer *request_body = evhttp_request_get_input_buffer(req);
  if (request_body != nullptr) {
    evbuffer_add_printf(response_body, "<br><br>====<br><br>\n");
    int result = evbuffer_add_buffer_reference(response_body, request_body);
    if (result < 0) {
      evbuffer_add_printf(response_body, ">>> Failed to print the body<br>\n");
    }
  }

  evhttp_add_header(evhttp_request_get_output_headers(req), "Content-Type",
                    "text/html");

  evhttp_send_reply(req, 200, "OK", response_body);

  evhttp_uri_free(decoded_url);
  free(decoded_path);
  evbuffer_free(response_body);
}

void help() { fprintf(stdout, "Usage: ev_print_req_server <port:8080>\n"); }

}  // namespace

int main(int argc, char **argv) {
  fprintf(stdout, "Start the http server ...\n");

  struct event_base *base;
  struct evhttp *http;
  struct evhttp_bound_socket *handle;

  ev_uint32_t port = 8080;

  if (argc < 2) {
    help();
    return 1;
  }

  bool port_parsed = absl::SimpleAtoi(argv[1], &port);
  if (!port_parsed) {
    fprintf(stderr, "Invalid port: %s\n", argv[1]);
  }

  base = event_base_new();
  if (!base) {
    fprintf(stderr, "Couldn't create an event_base: exiting\n");
    return 1;
  }

  http = evhttp_new(base);
  if (!http) {
    fprintf(stderr, "couldn't create evhttp. Exiting.\n");
    return 1;
  }

  // catch all
  evhttp_set_gencb(http, request_cb, NULL);

  // nullptr will bind to ipv4, which will fail to accept
  // requests from clients where getaddressinfo() defaults to AF_INET6
  handle = evhttp_bind_socket_with_handle(http, "::0", (ev_uint16_t)port);
  if (!handle) {
    fprintf(stderr, "couldn't bind to port %d. Exiting.\n", (int)port);
    return 1;
  }

  {
    /* Extract and display the address we're listening on. */
    struct sockaddr_storage ss = {};
    evutil_socket_t fd;
    ev_socklen_t socklen = sizeof(ss);
    char addrbuf[128];
    void *inaddr;
    const char *addr;
    int got_port = -1;
    fd = evhttp_bound_socket_get_fd(handle);
    memset(&ss, 0, sizeof(ss));
    if (getsockname(fd, (struct sockaddr *)&ss, &socklen)) {
      perror("getsockname() failed");
      return 1;
    }
    if (ss.ss_family == AF_INET) {
      got_port = ntohs(((struct sockaddr_in *)&ss)->sin_port);
      inaddr = &((struct sockaddr_in *)&ss)->sin_addr;
    } else if (ss.ss_family == AF_INET6) {
      got_port = ntohs(((struct sockaddr_in6 *)&ss)->sin6_port);
      inaddr = &((struct sockaddr_in6 *)&ss)->sin6_addr;
    } else {
      fprintf(stderr, "Weird address family %d\n", ss.ss_family);
      return 1;
    }
    addr = evutil_inet_ntop(ss.ss_family, inaddr, addrbuf, sizeof(addrbuf));
    if (addr) {
      printf("Listening on %s:%d\n", addr, got_port);
      evutil_snprintf(uri_root, sizeof(uri_root), "http://%s:%d", addr,
                      got_port);
    } else {
      fprintf(stderr, "evutil_inet_ntop failed\n");
      return 1;
    }
  }

  event_base_dispatch(base);

  return 0;
}
