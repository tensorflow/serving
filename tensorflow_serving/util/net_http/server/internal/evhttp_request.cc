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

#include "tensorflow_serving/util/net_http/server/internal/evhttp_request.h"

#include <zlib.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/base/internal/raw_logging.h"
#include "absl/strings/string_view.h"

#include "libevent/include/event2/buffer.h"
#include "libevent/include/event2/event.h"
#include "libevent/include/event2/http.h"
#include "libevent/include/event2/keyvalq_struct.h"

#include "tensorflow_serving/util/net_http/compression/gzip_zlib.h"
#include "tensorflow_serving/util/net_http/server/public/header_names.h"

namespace tensorflow {
namespace serving {
namespace net_http {

ParsedEvRequest::~ParsedEvRequest() {
  if (decoded_uri) {
    evhttp_uri_free(decoded_uri);
  }

  if (request && evhttp_request_is_owned(request)) {
    evhttp_request_free(request);
  }
}

ParsedEvRequest::ParsedEvRequest(evhttp_request* request_in)
    : request(request_in) {}

bool ParsedEvRequest::decode() {
  switch (evhttp_request_get_command(request)) {
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
      return false;
  }

  uri = evhttp_request_get_uri(request);

  decoded_uri = evhttp_uri_parse(uri);
  if (decoded_uri == nullptr) {
    return false;
  }

  // NB: need double-check "/" is OK
  path = evhttp_uri_get_path(decoded_uri);
  if (path == nullptr) {
    path = "/";
  }

  headers = evhttp_request_get_input_headers(request);

  return true;
}

EvHTTPRequest::EvHTTPRequest(std::unique_ptr<ParsedEvRequest> request,
                             ServerSupport* server)
    : server_(server),
      parsed_request_(std::move(request)),
      output_buf(nullptr) {}

EvHTTPRequest::~EvHTTPRequest() {
  if (output_buf != nullptr) {
    evbuffer_free(output_buf);
  }
}

absl::string_view EvHTTPRequest::uri_path() const {
  return parsed_request_->path;
}

absl::string_view EvHTTPRequest::http_method() const {
  return parsed_request_->method;
}

bool EvHTTPRequest::Initialize() {
  output_buf = evbuffer_new();
  return output_buf != nullptr;
}

void EvHTTPRequest::WriteResponseBytes(const char* data, int64_t size) {
  assert(size >= 0);
  if (output_buf == nullptr) {
    ABSL_RAW_LOG(FATAL, "Request not initialized.");
    return;
  }

  int ret = evbuffer_add(output_buf, data, static_cast<size_t>(size));
  if (ret == -1) {
    ABSL_RAW_LOG(ERROR, "Failed to write %zu bytes data to output buffer",
                 static_cast<size_t>(size));
  }
}

void EvHTTPRequest::WriteResponseString(absl::string_view data) {
  WriteResponseBytes(data.data(), static_cast<int64_t>(data.size()));
}

std::unique_ptr<char[], BlockDeleter> EvHTTPRequest::ReadRequestBytes(
    int64_t* size) {
  evbuffer* input_buf =
      evhttp_request_get_input_buffer(parsed_request_->request);
  if (input_buf == nullptr) {
    return nullptr;  // no body
  }

  auto buf_size = reinterpret_cast<size_t*>(size);

  *buf_size = evbuffer_get_contiguous_space(input_buf);

  if (*buf_size == 0) {
    return nullptr;  // EOF
  }

  char* block = std::allocator<char>().allocate(*buf_size);
  int ret = evbuffer_remove(input_buf, block, *buf_size);

  if (ret != *buf_size) {
    ABSL_RAW_LOG(ERROR, "Unexpected: read less than specified num_bytes : %zu",
                 *buf_size);
    std::allocator<char>().deallocate(block, *buf_size);
    *buf_size = 0;
    return nullptr;  // don't return corrupted buffer
  }

  // Uncompress the entire body
  if (NeedUncompressGzipContent()) {
    char* new_block;
    size_t orig_size = *buf_size;
    UncompressGzipContent(block, orig_size,
                          reinterpret_cast<void**>(&new_block), buf_size);
    std::allocator<char>().deallocate(block, orig_size);
    if (new_block != nullptr) {
      block = new_block;
    } else {
      ABSL_RAW_LOG(ERROR, "Failed to uncompress the gzipped body");
      *buf_size = 0;
      return nullptr;  // don't return corrupted buffer
    }
  }

  return std::unique_ptr<char[], BlockDeleter>(block, BlockDeleter(*buf_size));
}

bool EvHTTPRequest::NeedUncompressGzipContent() {
  if (handler_options_ != nullptr &&
      handler_options_->auto_uncompress_input()) {
    auto content_encoding = GetRequestHeader(HTTPHeaders::CONTENT_ENCODING);
    if (content_encoding != nullptr) {
      return content_encoding.find("gzip") != absl::string_view::npos;
    }
  }

  return false;
}

void EvHTTPRequest::UncompressGzipContent(void* input, size_t input_size,
                                          void** uncompressed_input,
                                          size_t* uncompressed_input_size) {
  int64_t max = handler_options_->auto_uncompress_max_size() > 0
                    ? handler_options_->auto_uncompress_max_size()
                    : ZLib::kMaxUncompressedBytes;

  // our APIs don't need expose the actual content-length
  *uncompressed_input_size = static_cast<size_t>(max);

  ZLib zlib;
  int err = zlib.UncompressGzipAndAllocate(
      reinterpret_cast<Bytef**>(uncompressed_input),
      reinterpret_cast<uLongf*>(uncompressed_input_size),
      static_cast<Bytef*>(input), static_cast<uLong>(input_size));
  if (err != Z_OK) {
    ABSL_RAW_LOG(ERROR, "Got zlib error: %d", err);
  }
}

// Note: passing string_view incurs a copy of underlying std::string data
// (stack)
absl::string_view EvHTTPRequest::GetRequestHeader(
    absl::string_view header) const {
  std::string header_str(header.data(), header.size());
  return absl::string_view(
      evhttp_find_header(parsed_request_->headers, header_str.c_str()));
}

std::vector<absl::string_view> EvHTTPRequest::request_headers() const {
  auto result = std::vector<absl::string_view>();
  auto ev_headers = parsed_request_->headers;

  for (evkeyval* header = ev_headers->tqh_first; header;
       header = header->next.tqe_next) {
    result.emplace_back(header->key);
  }

  return result;
}

void EvHTTPRequest::OverwriteResponseHeader(absl::string_view header,
                                            absl::string_view value) {
  evkeyvalq* ev_headers =
      evhttp_request_get_output_headers(parsed_request_->request);

  std::string header_str = std::string(header.data(), header.size());
  const char* header_cstr = header_str.c_str();

  evhttp_remove_header(ev_headers, header_cstr);
  evhttp_add_header(ev_headers, header_cstr,
                    std::string(value.data(), value.size()).c_str());
}

void EvHTTPRequest::AppendResponseHeader(absl::string_view header,
                                         absl::string_view value) {
  evkeyvalq* ev_headers =
      evhttp_request_get_output_headers(parsed_request_->request);

  int ret = evhttp_add_header(ev_headers,
                              std::string(header.data(), header.size()).c_str(),
                              std::string(value.data(), value.size()).c_str());

  if (ret != 0) {
    ABSL_RAW_LOG(ERROR,
                 "Unexpected: failed to set the request header"
                 " %.*s: %.*s",
                 static_cast<int>(header.size()), header.data(),
                 static_cast<int>(value.size()), value.data());
  }
}

void EvHTTPRequest::PartialReplyWithStatus(HTTPStatusCode status) {
  ABSL_RAW_LOG(FATAL, "PartialReplyWithStatus not implemented.");
}

void EvHTTPRequest::PartialReply() {
  ABSL_RAW_LOG(FATAL, "PartialReplyWithStatus not implemented.");
}

void EvHTTPRequest::ReplyWithStatus(HTTPStatusCode status) {
  bool result =
      server_->EventLoopSchedule([this, status]() { EvSendReply(status); });

  if (!result) {
    ABSL_RAW_LOG(ERROR, "Failed to EventLoopSchedule ReplyWithStatus()");
    Abort();
    // TODO(wenboz): should have a forced abort that doesn't write back anything
    // to the event-loop
  }
}

void EvHTTPRequest::EvSendReply(HTTPStatusCode status) {
  evhttp_send_reply(parsed_request_->request, static_cast<int>(status), nullptr,
                    output_buf);
  server_->DecOps();
  delete this;
}

void EvHTTPRequest::Reply() { ReplyWithStatus(HTTPStatusCode::OK); }

// Treats this as 500 for now and let libevent decide what to do
// with the connection.
void EvHTTPRequest::Abort() {
  evhttp_send_error(parsed_request_->request, HTTP_INTERNAL, nullptr);
  server_->DecOps();
  delete this;
}

}  // namespace net_http
}  // namespace serving
}  // namespace tensorflow
