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

#include "tensorflow_serving/model_servers/http_server.h"

#include <cstdint>
#include <memory>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "re2/re2.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow_serving/model_servers/http_rest_api_handler.h"
#include "tensorflow_serving/model_servers/http_rest_api_util.h"
#include "tensorflow_serving/model_servers/server_core.h"
#include "tensorflow_serving/model_servers/server_init.h"
#include "tensorflow_serving/servables/tensorflow/util.h"
#include "tensorflow_serving/util/net_http/public/response_code_enum.h"
#include "tensorflow_serving/util/net_http/server/public/httpserver.h"
#include "tensorflow_serving/util/net_http/server/public/server_request_interface.h"
#include "tensorflow_serving/util/prometheus_exporter.h"
#include "tensorflow_serving/util/threadpool_executor.h"

namespace tensorflow {
namespace serving {

namespace {

net_http::HTTPStatusCode ToHTTPStatusCode(const Status& status) {
  using error::Code;
  using net_http::HTTPStatusCode;
  switch (static_cast<absl::StatusCode>(status.code())) {
    case absl::StatusCode::kOk:
      return HTTPStatusCode::OK;
    case absl::StatusCode::kCancelled:
      return HTTPStatusCode::CLIENT_CLOSED_REQUEST;
    case absl::StatusCode::kUnknown:
      return HTTPStatusCode::ERROR;
    case absl::StatusCode::kInvalidArgument:
      return HTTPStatusCode::BAD_REQUEST;
    case absl::StatusCode::kDeadlineExceeded:
      return HTTPStatusCode::GATEWAY_TO;
    case absl::StatusCode::kNotFound:
      return HTTPStatusCode::NOT_FOUND;
    case absl::StatusCode::kAlreadyExists:
      return HTTPStatusCode::CONFLICT;
    case absl::StatusCode::kPermissionDenied:
      return HTTPStatusCode::FORBIDDEN;
    case absl::StatusCode::kResourceExhausted:
      return HTTPStatusCode::TOO_MANY_REQUESTS;
    case absl::StatusCode::kFailedPrecondition:
      return HTTPStatusCode::BAD_REQUEST;
    case absl::StatusCode::kAborted:
      return HTTPStatusCode::CONFLICT;
    case absl::StatusCode::kOutOfRange:
      return HTTPStatusCode::BAD_REQUEST;
    case absl::StatusCode::kUnimplemented:
      return HTTPStatusCode::NOT_IMP;
    case absl::StatusCode::kInternal:
      return HTTPStatusCode::ERROR;
    case absl::StatusCode::kUnavailable:
      return HTTPStatusCode::SERVICE_UNAV;
    case absl::StatusCode::kDataLoss:
      return HTTPStatusCode::ERROR;
    case absl::StatusCode::kUnauthenticated:
      return HTTPStatusCode::UNAUTHORIZED;
    default:
      return HTTPStatusCode::ERROR;
  }
}

void ProcessPrometheusRequest(PrometheusExporter* exporter, const string& path,
                              net_http::ServerRequestInterface* req) {
  std::vector<std::pair<string, string>> headers;
  headers.push_back({"Content-Type", "text/plain"});
  string output;
  Status status;
  // Check if url matches the path.
  if (req->uri_path() != path) {
    output = absl::StrFormat("Unexpected path: %s. Should be %s",
                             req->uri_path(), path);
    status = Status(static_cast<tensorflow::errors::Code>(
                        absl::StatusCode::kInvalidArgument),
                    output);
  } else {
    status = exporter->GeneratePage(&output);
  }
  const net_http::HTTPStatusCode http_status = ToHTTPStatusCode(status);
  // Note: we add headers+output for non successful status too, in case the
  // output contains details about the error (e.g. error messages).
  for (const auto& kv : headers) {
    req->OverwriteResponseHeader(kv.first, kv.second);
  }
  req->WriteResponseString(output);
  if (http_status != net_http::HTTPStatusCode::OK) {
    VLOG(1) << "Error Processing prometheus metrics request. Error: "
            << status.ToString();
  }
  req->ReplyWithStatus(http_status);
}

class RequestExecutor final : public net_http::EventExecutor {
 public:
  explicit RequestExecutor(int num_threads)
      : executor_(Env::Default(), "httprestserver", num_threads) {}

  void Schedule(std::function<void()> fn) override { executor_.Schedule(fn); }

 private:
  ThreadPoolExecutor executor_;
};

class RestApiRequestDispatcher {
 public:
  RestApiRequestDispatcher(int timeout_in_ms, ServerCore* core)
      : regex_(HttpRestApiHandler::kPathRegex),
        core_(core),
        handler_(tensorflow::serving::init::CreateHttpRestApiHandler(
            timeout_in_ms, core)) {}

  net_http::RequestHandler Dispatch(net_http::ServerRequestInterface* req) {
    if (RE2::FullMatch(string(req->uri_path()), regex_)) {
      return [this](net_http::ServerRequestInterface* req) {
        this->ProcessRequest(req);
      };
    }
    VLOG(1) << "Ignoring HTTP request: " << req->http_method() << " "
            << req->uri_path();
    return nullptr;
  }

 private:
  void ProcessRequest(net_http::ServerRequestInterface* req) {
    const uint64_t start = Env::Default()->NowMicros();
    string body;
    int64_t num_bytes = 0;
    auto request_chunk = req->ReadRequestBytes(&num_bytes);
    while (request_chunk != nullptr) {
      absl::StrAppend(&body, absl::string_view(request_chunk.get(), num_bytes));
      request_chunk = req->ReadRequestBytes(&num_bytes);
    }

    std::vector<std::pair<string, string>> headers;
    string model_name;
    string method;
    string output;
    VLOG(1) << "Processing HTTP request: " << req->http_method() << " "
            << req->uri_path() << " body: " << body.size() << " bytes.";

    Status status;
    if (req->http_method() == "OPTIONS") {
      absl::string_view origin_header = req->GetRequestHeader("Origin");
      if (RE2::PartialMatch(origin_header, "https?://")) {
        status = OkStatus();
      } else {
        status = errors::FailedPrecondition(
            "Origin header is missing in CORS preflight");
      }
    } else {
      status =
          handler_->ProcessRequest(req->http_method(), req->uri_path(), body,
                                   &headers, &model_name, &method, &output);
    }
    if (core_->enable_cors_support()) {
      AddCORSHeaders(&headers);
    }

    const auto http_status = ToHTTPStatusCode(status);
    // Note: we add headers+output for non successful status too, in case the
    // output contains details about the error (e.g. error messages).
    for (const auto& kv : headers) {
      req->OverwriteResponseHeader(kv.first, kv.second);
    }
    req->WriteResponseString(output);
    if (http_status == net_http::HTTPStatusCode::OK) {
      RecordRequestLatency(model_name, /*api=*/method, /*entrypoint=*/"REST",
                           Env::Default()->NowMicros() - start);
    } else {
      VLOG(1) << "Error Processing HTTP/REST request: " << req->http_method()
              << " " << req->uri_path() << " Error: " << status.ToString();
    }
    RecordModelRequestCount(model_name, status);
    req->ReplyWithStatus(http_status);
  }

  const RE2 regex_;
  ServerCore* core_;
  std::unique_ptr<HttpRestApiHandlerBase> handler_;
};

}  // namespace

std::unique_ptr<net_http::HTTPServerInterface> CreateAndStartHttpServer(
    int port, int num_threads, int timeout_in_ms,
    const MonitoringConfig& monitoring_config, ServerCore* core) {
  auto options = absl::make_unique<net_http::ServerOptions>();
  options->AddPort(static_cast<uint32_t>(port));
  options->SetExecutor(absl::make_unique<RequestExecutor>(num_threads));

  auto server = net_http::CreateEvHTTPServer(std::move(options));
  if (server == nullptr) {
    return nullptr;
  }

  // Register handler for prometheus metric endpoint.
  if (monitoring_config.prometheus_config().enable()) {
    std::shared_ptr<PrometheusExporter> exporter =
        std::make_shared<PrometheusExporter>();
    net_http::RequestHandlerOptions prometheus_request_options;
    PrometheusConfig prometheus_config = monitoring_config.prometheus_config();
    auto path = prometheus_config.path().empty()
                    ? PrometheusExporter::kPrometheusPath
                    : prometheus_config.path();
    server->RegisterRequestHandler(
        path,
        [exporter, path](net_http::ServerRequestInterface* req) {
          ProcessPrometheusRequest(exporter.get(), path, req);
        },
        prometheus_request_options);
  }

  std::shared_ptr<RestApiRequestDispatcher> dispatcher =
      std::make_shared<RestApiRequestDispatcher>(timeout_in_ms, core);
  net_http::RequestHandlerOptions handler_options;
  server->RegisterRequestDispatcher(
      [dispatcher](net_http::ServerRequestInterface* req) {
        return dispatcher->Dispatch(req);
      },
      handler_options);
  if (server->StartAcceptingRequests()) {
    return server;
  }
  return nullptr;
}

}  // namespace serving
}  // namespace tensorflow
