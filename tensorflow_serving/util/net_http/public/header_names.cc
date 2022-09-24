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

#include "tensorflow_serving/util/net_http/public/header_names.h"

namespace tensorflow {
namespace serving {
namespace net_http {

// Standard header names

const char HTTPHeaders::ACCEPT[] = "Accept";
const char HTTPHeaders::ACCEPT_CHARSET[] = "Accept-Charset";
const char HTTPHeaders::ACCEPT_ENCODING[] = "Accept-Encoding";
const char HTTPHeaders::ACCEPT_LANGUAGE[] = "Accept-Language";
const char HTTPHeaders::ACCEPT_RANGES[] = "Accept-Ranges";
const char HTTPHeaders::ACCESS_CONTROL_ALLOW_CREDENTIALS[] =
    "Access-Control-Allow-Credentials";
const char HTTPHeaders::ACCESS_CONTROL_ALLOW_HEADERS[] =
    "Access-Control-Allow-Headers";
const char HTTPHeaders::ACCESS_CONTROL_ALLOW_METHODS[] =
    "Access-Control-Allow-Methods";
const char HTTPHeaders::ACCESS_CONTROL_ALLOW_ORIGIN[] =
    "Access-Control-Allow-Origin";
const char HTTPHeaders::ACCESS_CONTROL_EXPOSE_HEADERS[] =
    "Access-Control-Expose-Headers";
const char HTTPHeaders::ACCESS_CONTROL_MAX_AGE[] = "Access-Control-Max-Age";
const char HTTPHeaders::ACCESS_CONTROL_REQUEST_HEADERS[] =
    "Access-Control-Request-Headers";
const char HTTPHeaders::ACCESS_CONTROL_REQUEST_METHOD[] =
    "Access-Control-Request-Method";
const char HTTPHeaders::AGE[] = "Age";
const char HTTPHeaders::ALLOW[] = "Allow";
const char HTTPHeaders::AUTHORIZATION[] = "Authorization";
const char HTTPHeaders::CACHE_CONTROL[] = "Cache-Control";
const char HTTPHeaders::CONNECTION[] = "Connection";
const char HTTPHeaders::CONTENT_DISPOSITION[] = "Content-Disposition";
const char HTTPHeaders::CONTENT_ENCODING[] = "Content-Encoding";
const char HTTPHeaders::CONTENT_LANGUAGE[] = "Content-Language";
const char HTTPHeaders::CONTENT_LENGTH[] = "Content-Length";
const char HTTPHeaders::CONTENT_LOCATION[] = "Content-Location";
const char HTTPHeaders::CONTENT_RANGE[] = "Content-Range";
const char HTTPHeaders::CONTENT_SECURITY_POLICY[] = "Content-Security-Policy";
const char HTTPHeaders::CONTENT_SECURITY_POLICY_REPORT_ONLY[] =
    "Content-Security-Policy-Report-Only";
const char HTTPHeaders::X_CONTENT_SECURITY_POLICY[] =
    "X-Content-Security-Policy";
const char HTTPHeaders::X_CONTENT_SECURITY_POLICY_REPORT_ONLY[] =
    "X-Content-Security-Policy-Report-Only";
const char HTTPHeaders::X_WEBKIT_CSP[] = "X-WebKit-CSP";
const char HTTPHeaders::X_WEBKIT_CSP_REPORT_ONLY[] = "X-WebKit-CSP-Report-Only";
const char HTTPHeaders::CONTENT_TYPE[] = "Content-Type";
const char HTTPHeaders::CONTENT_MD5[] = "Content-MD5";
const char HTTPHeaders::X_CONTENT_TYPE_OPTIONS[] = "X-Content-Type-Options";
const char HTTPHeaders::COOKIE[] = "Cookie";
const char HTTPHeaders::COOKIE2[] = "Cookie2";
const char HTTPHeaders::DATE[] = "Date";
const char HTTPHeaders::DAV[] = "DAV";
const char HTTPHeaders::DEPTH[] = "Depth";
const char HTTPHeaders::DESTINATION[] = "Destination";
const char HTTPHeaders::DNT[] = "DNT";
const char HTTPHeaders::EARLY_DATA[] = "Early-Data";
const char HTTPHeaders::ETAG[] = "ETag";
const char HTTPHeaders::EXPECT[] = "Expect";
const char HTTPHeaders::EXPIRES[] = "Expires";
const char HTTPHeaders::FOLLOW_ONLY_WHEN_PRERENDER_SHOWN[] =
    "Follow-Only-When-Prerender-Shown";
const char HTTPHeaders::FORWARDED[] = "Forwarded";
const char HTTPHeaders::FROM[] = "From";
const char HTTPHeaders::HOST[] = "Host";
const char HTTPHeaders::HTTP2_SETTINGS[] = "HTTP2-Settings";
const char HTTPHeaders::IF[] = "If";
const char HTTPHeaders::IF_MATCH[] = "If-Match";
const char HTTPHeaders::IF_MODIFIED_SINCE[] = "If-Modified-Since";
const char HTTPHeaders::IF_NONE_MATCH[] = "If-None-Match";
const char HTTPHeaders::IF_UNMODIFIED_SINCE[] = "If-Unmodified-Since";
const char HTTPHeaders::IF_RANGE[] = "If-Range";
const char HTTPHeaders::KEEP_ALIVE[] = "Keep-Alive";
const char HTTPHeaders::LABEL[] = "Label";
const char HTTPHeaders::LAST_MODIFIED[] = "Last-Modified";
const char HTTPHeaders::LINK[] = "Link";
const char HTTPHeaders::LOCATION[] = "Location";
const char HTTPHeaders::LOCK_TOKEN[] = "Lock-Token";
const char HTTPHeaders::MAX_FORWARDS[] = "Max-Forwards";
const char HTTPHeaders::MS_AUTHOR_VIA[] = "MS-Author-Via";
const char HTTPHeaders::ORIGIN[] = "Origin";
const char HTTPHeaders::OVERWRITE_HDR[] = "Overwrite";
const char HTTPHeaders::PRAGMA[] = "Pragma";
const char HTTPHeaders::P3P[] = "P3P";
const char HTTPHeaders::PING_FROM[] = "Ping-From";
const char HTTPHeaders::PING_TO[] = "Ping-To";
const char HTTPHeaders::PROXY_CONNECTION[] = "Proxy-Connection";
const char HTTPHeaders::PROXY_AUTHENTICATE[] = "Proxy-Authenticate";
const char HTTPHeaders::PROXY_AUTHORIZATION[] = "Proxy-Authorization";
const char HTTPHeaders::PUBLIC_KEY_PINS[] = "Public-Key-Pins";
const char HTTPHeaders::PUBLIC_KEY_PINS_REPORT_ONLY[] =
    "Public-Key-Pins-Report-Only";
const char HTTPHeaders::RANGE[] = "Range";
const char HTTPHeaders::REFERER[] = "Referer";
const char HTTPHeaders::REFERRER_POLICY[] = "Referrer-Policy";
const char HTTPHeaders::REFERRER_POLICY_NO_REFERRER[] = "no-referrer";
const char HTTPHeaders::REFERRER_POLICY_NO_REFFERER_WHEN_DOWNGRADE[] =
    "no-referrer-when-downgrade";
const char HTTPHeaders::REFERRER_POLICY_SAME_ORIGIN[] = "same-origin";
const char HTTPHeaders::REFERRER_POLICY_ORIGIN[] = "origin";
const char HTTPHeaders::REFERRER_POLICY_STRICT_ORIGIN[] = "strict-origin";
const char HTTPHeaders::REFERRER_POLICY_ORIGIN_WHEN_CROSS_ORIGIN[] =
    "origin-when-cross-origin";
const char HTTPHeaders::REFERRER_POLICY_STRICT_ORIGIN_WHEN_CROSS_ORIGIN[] =
    "strict-origin-when-cross-origin";
const char HTTPHeaders::REFERRER_POLICY_UNSAFE_URL[] = "unsafe-url";
const char HTTPHeaders::REFRESH[] = "Refresh";
const char HTTPHeaders::RETRY_AFTER[] = "Retry-After";
const char HTTPHeaders::SEC_METADATA[] = "Sec-Metadata";
const char HTTPHeaders::SEC_TOKEN_BINDING[] = "Sec-Token-Binding";
const char HTTPHeaders::SEC_PROVIDED_TOKEN_BINDING_ID[] =
    "Sec-Provided-Token-Binding-ID";
const char HTTPHeaders::SEC_REFERRED_TOKEN_BINDING_ID[] =
    "Sec-Referred-Token-Binding-ID";
const char HTTPHeaders::SERVER[] = "Server";
const char HTTPHeaders::SERVER_TIMING[] = "Server-Timing";
const char HTTPHeaders::SERVICE_WORKER[] = "Service-Worker";
const char HTTPHeaders::SERVICE_WORKER_ALLOWED[] = "Service-Worker-Allowed";
const char HTTPHeaders::SERVICE_WORKER_NAVIGATION_PRELOAD[] =
    "Service-Worker-Navigation-Preload";
const char HTTPHeaders::SET_COOKIE[] = "Set-Cookie";
const char HTTPHeaders::SET_COOKIE2[] = "Set-Cookie2";
const char HTTPHeaders::STATUS_URI[] = "Status-URI";
const char HTTPHeaders::STRICT_TRANSPORT_SECURITY[] =
    "Strict-Transport-Security";
const char HTTPHeaders::TIMEOUT[] = "Timeout";
const char HTTPHeaders::TIMING_ALLOW_ORIGIN[] = "Timing-Allow-Origin";
const char HTTPHeaders::TK[] = "Tk";
const char HTTPHeaders::TRAILER[] = "Trailer";
const char HTTPHeaders::TRAILERS[] = "Trailers";
const char HTTPHeaders::TRANSFER_ENCODING[] = "Transfer-Encoding";
const char HTTPHeaders::TRANSFER_ENCODING_ABBRV[] = "TE";
const char HTTPHeaders::UPGRADE[] = "Upgrade";
const char HTTPHeaders::USER_AGENT[] = "User-Agent";
const char HTTPHeaders::VARY[] = "Vary";
const char HTTPHeaders::VIA[] = "Via";
const char HTTPHeaders::WARNING[] = "Warning";
const char HTTPHeaders::WWW_AUTHENTICATE[] = "WWW-Authenticate";

}  // namespace net_http
}  // namespace serving
}  // namespace tensorflow
