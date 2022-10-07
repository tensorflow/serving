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

// net_http::ServerRequestInterface defines a pure interface class for handling
// an HTTP request on the server-side. It is designed as a minimum API
// to ensure reusability and to work with different HTTP implementations.
//
// ServerRequestInterface is thread-compatible. Once the request object
// has been dispatched to the application-specified handler, the HTTP server
// runtime will not access the request object till Reply() is called.
//
// Streamed request/response APIs are to be added, which will introduce
// additional API contract wrt the threading semantics.

#ifndef TENSORFLOW_SERVING_UTIL_NET_HTTP_SERVER_PUBLIC_SERVER_REQUEST_INTERFACE_H_
#define TENSORFLOW_SERVING_UTIL_NET_HTTP_SERVER_PUBLIC_SERVER_REQUEST_INTERFACE_H_

#include <cstdlib>
#include <functional>
#include <memory>
#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow_serving/util/net_http/public/response_code_enum.h"

namespace tensorflow {
namespace serving {
namespace net_http {

class ServerRequestInterface {
 public:
  // To be used with memory blocks returned via std::unique_ptr<char[]>
  struct BlockDeleter {
   public:
    BlockDeleter() : size_(0) {}  // nullptr
    explicit BlockDeleter(int64_t size) : size_(size) {}
    inline void operator()(char* ptr) const {
      // TODO: c++14 ::operator delete[](ptr, size_t)
      std::allocator<char>().deallocate(ptr, static_cast<std::size_t>(size_));
    }

   private:
    int64_t size_;
  };

  virtual ~ServerRequestInterface() = default;

  ServerRequestInterface(const ServerRequestInterface& other) = delete;
  ServerRequestInterface& operator=(const ServerRequestInterface& other) =
      delete;

  // The portion of the request URI after the host and port.
  // E.g. "/path/to/resource?param=value&param=value#fragment".
  // Doesn't unescape the contents; returns "/" at least.
  virtual absl::string_view uri_path() const = 0;

  // HTTP request method.
  // Must be in Upper Case.
  virtual absl::string_view http_method() const = 0;

  // Input/output byte-buffer types are subject to change!
  // I/O buffer choices:
  // - absl::ByteStream would work but it is not yet open-sourced
  // - iovec doesn't add much value and may limit portability; but otherwise
  //   the current API is compatible with iovec
  // - absl::Span is open-sourced, but string_view is simpler to use for writing
  //   to an HTTP response.

  // Appends the data block of the specified size to the response body.
  // This request object takes the ownership of the data block.
  //
  // Note this is not a streaming write API. See PartialReply() below.
  virtual void WriteResponseBytes(const char* data, int64_t size) = 0;

  // Appends (by coping) the data of string_view to the end of
  // the response body.
  virtual void WriteResponseString(absl::string_view data) = 0;

  // Reads from the request body.
  // Returns the number bytes of data read, whose ownership will be transferred
  // to the caller. Returns nullptr when EOF is reached or when there
  // is no request body.
  //
  // The returned memory will be "free-ed" via the custom Deleter. Do not
  // release the memory manually as its allocator is subject to change.
  //
  // Note this is not a streaming read API in that the complete request body
  // should have already been received.
  virtual std::unique_ptr<char[], ServerRequestInterface::BlockDeleter>
  ReadRequestBytes(int64_t* size) = 0;

  // Returns the first value, including "", associated with a request
  // header name. The header name argument is case-insensitive.
  // Returns nullptr if the specified header doesn't exist.
  virtual absl::string_view GetRequestHeader(
      absl::string_view header) const = 0;

  // To be added: multi-value headers.

  // Returns all the request header names.
  // This is not an efficient way to access headers, mainly for debugging uses.
  virtual std::vector<absl::string_view> request_headers() const = 0;

  virtual void OverwriteResponseHeader(absl::string_view header,
                                       absl::string_view value) = 0;
  virtual void AppendResponseHeader(absl::string_view header,
                                    absl::string_view value) = 0;

  // The IO status of a request or response body.
  enum class BodyStatus {
    // The body hasn't been completely read or written.
    PENDING = 0,
    // The body has been completely read or written or when there is no body.
    COMPLETE = 1,
    // The transport has reported a failure and the request should be aborted.
    FAILED = 2,
  };

  // This serves as the return value type for callbacks that may be
  // skipped for optimization reasons
  enum class CallbackStatus {
    NOT_SCHEDULED = 0,
    SCHEDULED = 1,
  };

  // Sends headers and/or any buffered response body data to the client.
  // Assumes 200 if status is not specified.
  // If called for the first time, all the response headers will be sent
  // together including headers specified by the application and
  // headers generated by the server.
  // Trying to modify headers or specifying a status after the first
  // PartialReply() is called is considered a programming error and
  // the underlying behavior is undefined.
  virtual void PartialReplyWithStatus(HTTPStatusCode status) = 0;
  virtual void PartialReply() = 0;

  // Similar to PartialReply() but with an on_flush callback which will be
  // invoked when the response data has been completely flushed by the
  // transport. This allows the handler to apply transport-provided flow-control
  // in writing data to the peer.
  //
  // Returns SCHEDULED if the callback will be invoked asynchronously after
  // this method returns. Until the callback is invoked, the request object
  // should not be accessed by the handler.
  //
  // Returns NOT_SCHEDULED if data is already flushed when this method returns
  // or when the request should be aborted due to transport failures.
  //
  // The handler should check response_body_status() after this method returns
  // or from the callback to decide if the request should be aborted due to
  // transport failures.
  virtual CallbackStatus PartialReplyWithFlushCallback(
      std::function<void()> callback) = 0;
  virtual BodyStatus response_body_status() { return BodyStatus::PENDING; }

  // Request streaming is disabled by default
  virtual BodyStatus request_body_status() { return BodyStatus::COMPLETE; }

  // Completes the response and sends any buffered response body
  // to the client. Headers will be generated and sent first if PartialReply()
  // has never be called.
  // Assumes 200 if status is not specified.
  // Once Reply() is called, the request object will be owned and destructed
  // by the server runtime.
  virtual void ReplyWithStatus(HTTPStatusCode status) = 0;
  virtual void Reply() = 0;

  // Aborts the current request forcibly.
  // Once Abort() is called, the request object will be owned and destructed
  // by the server runtime.
  virtual void Abort() = 0;

 protected:
  ServerRequestInterface() = default;

 private:
  // Do not add any data members to this class.
};

// Helper methods.

inline void SetContentType(ServerRequestInterface* request,
                           absl::string_view type) {
  request->OverwriteResponseHeader("Content-Type", type);
}

inline void SetContentTypeHTML(ServerRequestInterface* request) {
  SetContentType(request, "text/html");
}

inline void SetContentTypeTEXT(ServerRequestInterface* request) {
  SetContentType(request, "text/plain");
}

}  // namespace net_http
}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_UTIL_NET_HTTP_SERVER_PUBLIC_SERVER_REQUEST_INTERFACE_H_
