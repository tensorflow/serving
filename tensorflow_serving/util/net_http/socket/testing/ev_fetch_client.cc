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

#include <sys/types.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "libevent/include/event2/buffer.h"
#include "libevent/include/event2/bufferevent.h"
#include "libevent/include/event2/event.h"
#include "libevent/include/event2/http.h"
#include "libevent/include/event2/keyvalq_struct.h"
#include "libevent/include/event2/util.h"

namespace {

struct event_base* ev_base;
struct evhttp_uri* http_uri;
struct evhttp_connection* evcon;

void response_cb(struct evhttp_request* req, void* ctx) {
  char buffer[1024];
  int nread;

  if (req == nullptr) {
    int errcode = EVUTIL_SOCKET_ERROR();
    fprintf(stderr, "socket error = %s (%d)\n",
            evutil_socket_error_to_string(errcode), errcode);
    return;
  }

  fprintf(stderr, "Response line: %d %s\n",
          evhttp_request_get_response_code(req),
          evhttp_request_get_response_code_line(req));

  struct evkeyvalq* headers = evhttp_request_get_input_headers(req);
  struct evkeyval* header;
  for (header = headers->tqh_first; header; header = header->next.tqe_next) {
    fprintf(stderr, "%s : %s\n", header->key, header->value);
  }

  while ((nread = evbuffer_remove(evhttp_request_get_input_buffer(req), buffer,
                                  sizeof(buffer))) > 0) {
    fwrite(buffer, static_cast<size_t>(nread), 1, stderr);
  }
}

void help() { fputs("Usage: ev_fetch-client uri [body]\n", stderr); }

void err(const char* msg) { fputs(msg, stderr); }

void cleanup() {
  if (evcon != nullptr) {
    evhttp_connection_free(evcon);
  }
  if (http_uri != nullptr) {
    evhttp_uri_free(http_uri);
  }

  event_base_free(ev_base);
}

}  // namespace

int main(int argc, char** argv) {
  fprintf(stdout, "Start the http client ...\n");

  if (argc < 2 || argc > 3) {
    help();
    return 1;
  }

  const char* url = argv[1];

  const char* body = nullptr;

  if (argc == 3) {
    body = argv[2];
  }

  http_uri = evhttp_uri_parse(url);
  if (http_uri == nullptr) {
    err("malformed url");
    return 1;
  }

  const char* scheme = evhttp_uri_get_scheme(http_uri);
  if (scheme == nullptr || strcasecmp(scheme, "http") != 0) {
    err("url must be http");
    return 1;
  }

  const char* host = evhttp_uri_get_host(http_uri);
  if (host == nullptr) {
    err("url must have a host");
    return 1;
  }

  int port = evhttp_uri_get_port(http_uri);
  if (port == -1) {
    port = (strcasecmp(scheme, "http") == 0) ? 80 : 443;
  }

  const char* path = evhttp_uri_get_path(http_uri);
  if (strlen(path) == 0) {
    path = "/";
  }

  char uri[256];
  const char* query = evhttp_uri_get_query(http_uri);
  if (query == nullptr) {
    snprintf(uri, sizeof(uri) - 1, "%s", path);
  } else {
    snprintf(uri, sizeof(uri) - 1, "%s?%s", path, query);
  }
  uri[sizeof(uri) - 1] = '\0';

  // Create event base
  ev_base = event_base_new();
  if (ev_base == nullptr) {
    perror("event_base_new()");
    return 1;
  }

  //  struct bufferevent* bev =
  //      bufferevent_socket_new(ev_base, -1, BEV_OPT_CLOSE_ON_FREE);
  //
  //  if (bev == nullptr) {
  //    fprintf(stderr, "bufferevent_socket_new() failed\n");
  //    return 1;
  //  }

  // blocking call (DNS resolution)
  evcon = evhttp_connection_base_bufferevent_new(
      ev_base, nullptr, nullptr, host, static_cast<uint16_t>(port));
  if (evcon == nullptr) {
    fprintf(stderr, "evhttp_connection_base_bufferevent_new() failed\n");
    return 1;
  }

  int retries = 0;
  evhttp_connection_set_retries(evcon, retries);

  int timeout_in_secs = 5;
  evhttp_connection_set_timeout(evcon, timeout_in_secs);

  struct evhttp_request* req = evhttp_request_new(response_cb, evcon);
  if (req == nullptr) {
    fprintf(stderr, "evhttp_request_new() failed\n");
    return 1;
  }

  struct evkeyvalq* output_headers = evhttp_request_get_output_headers(req);
  evhttp_add_header(output_headers, "Host", host);
  evhttp_add_header(output_headers, "Connection", "close");

  if (body) {
    struct evbuffer* output_buffer = evhttp_request_get_output_buffer(req);
    size_t size = strlen(body);

    evbuffer_add(output_buffer, body, size);

    char length_header[8];
    evutil_snprintf(length_header, sizeof(length_header) - 1, "%lu", size);
    evhttp_add_header(output_headers, "Content-Length", length_header);
  }

  int r = evhttp_make_request(evcon, req,
                              body ? EVHTTP_REQ_POST : EVHTTP_REQ_GET, uri);
  if (r != 0) {
    fprintf(stderr, "evhttp_make_request() failed\n");
    return 1;
  }

  event_base_dispatch(ev_base);

  cleanup();

  return 0;
}
