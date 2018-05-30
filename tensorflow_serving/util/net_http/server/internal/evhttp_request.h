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

// libevent based request implementation

#ifndef TENSORFLOW_SERVING_UTIL_NET_HTTP_SERVER_INTERNAL_EVHTTP_REQUEST_H_
#define TENSORFLOW_SERVING_UTIL_NET_HTTP_SERVER_INTERNAL_EVHTTP_REQUEST_H_

#include <cstdint>
#include <memory>

#include "tensorflow_serving/util/net_http/server/internal/server_support.h"
#include "tensorflow_serving/util/net_http/server/public/serverrequestinterface.h"

struct evbuffer;
struct evhttp_request;
struct evhttp_uri;
struct evkeyvalq;

namespace tensorflow {
namespace serving {
namespace net_http {

// Headers only
struct ParsedEvRequest {
 public:
  // Doesn't take the ownership
  explicit ParsedEvRequest(evhttp_request* request_in);
  ~ParsedEvRequest();

  // Decode and cache the result; or return false if any parsing error
  bool decode();

  evhttp_request* request;  // raw request

  const char* method;  // from enum

  const char* uri = nullptr;          // from raw request
  evhttp_uri* decoded_uri = nullptr;  // owned by this

  // TODO(wenboz): do we need escaped path for dispatching requests?
  // evhttp_uridecode(path)
  const char* path = nullptr;  // owned by uri

  evkeyvalq* headers = nullptr;  // owned by raw request
};

// Thread-compatible. See ServerRequestInterface on the exact contract
// between the server runtime and application handlers.
class EvHTTPRequest final : public ServerRequestInterface {
 public:
  virtual ~EvHTTPRequest();

  EvHTTPRequest(const EvHTTPRequest& other) = delete;
  EvHTTPRequest& operator=(const EvHTTPRequest& other) = delete;

  // Doesn't own the server
  EvHTTPRequest(std::unique_ptr<ParsedEvRequest> request,
                ServerSupport* server);

  absl::string_view uri_path() const override;

  absl::string_view http_method() const override;

  void WriteResponseBytes(const char* data, int64_t size) override;

  void WriteResponseString(absl::string_view data) override;

  std::unique_ptr<char, FreeDeleter> ReadRequestBytes(int64_t* size) override;

  absl::string_view GetRequestHeader(absl::string_view header) const override;

  std::vector<absl::string_view> request_headers() const override;

  void OverwriteResponseHeader(absl::string_view header,
                               absl::string_view value) override;
  void AppendResponseHeader(absl::string_view header,
                            absl::string_view value) override;

  void PartialReplyWithStatus(HTTPStatusCode status) override;
  void PartialReply() override;

  void ReplyWithStatus(HTTPStatusCode status) override;
  void Reply() override;

  void Abort() override;

  // Initializes the resource and returns false if any error.
  bool Initialize();

 private:
  void EvSendReply(HTTPStatusCode status);

  ServerSupport* server_;

  std::unique_ptr<ParsedEvRequest> parsed_request_;

  evbuffer* output_buf;  // owned by this
};

}  // namespace net_http
}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_UTIL_NET_HTTP_SERVER_INTERNAL_EVHTTP_REQUEST_H_
