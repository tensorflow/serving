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

#ifndef TENSORFLOW_SERVING_UTIL_NET_HTTP_PUBLIC_HEADER_NAMES_H_
#define TENSORFLOW_SERVING_UTIL_NET_HTTP_PUBLIC_HEADER_NAMES_H_

namespace tensorflow {
namespace serving {
namespace net_http {

// Standard HTTP Header Names
//
// http://www.iana.org/assignments/message-headers
class HTTPHeaders {
 public:
  HTTPHeaders() = delete;

  static const char ACCEPT[];
  static const char ACCEPT_CHARSET[];
  static const char ACCEPT_ENCODING[];
  static const char ACCEPT_LANGUAGE[];
  static const char ACCEPT_RANGES[];
  static const char ACCESS_CONTROL_ALLOW_CREDENTIALS[];
  static const char ACCESS_CONTROL_ALLOW_HEADERS[];
  static const char ACCESS_CONTROL_ALLOW_METHODS[];
  static const char ACCESS_CONTROL_ALLOW_ORIGIN[];
  static const char ACCESS_CONTROL_EXPOSE_HEADERS[];
  static const char ACCESS_CONTROL_MAX_AGE[];
  static const char ACCESS_CONTROL_REQUEST_HEADERS[];
  static const char ACCESS_CONTROL_REQUEST_METHOD[];
  static const char AGE[];
  static const char ALLOW[];
  static const char AUTHORIZATION[];
  static const char CACHE_CONTROL[];
  static const char CONNECTION[];
  static const char CONTENT_DISPOSITION[];
  static const char CONTENT_ENCODING[];
  static const char CONTENT_LANGUAGE[];
  static const char CONTENT_LENGTH[];
  static const char CONTENT_LOCATION[];
  static const char CONTENT_RANGE[];
  // http://w3.org/TR/CSP/#content-security-policy-header-field
  static const char CONTENT_SECURITY_POLICY[];
  // http://w3.org/TR/CSP/#content-security-policy-report-only-header-field
  static const char CONTENT_SECURITY_POLICY_REPORT_ONLY[];
  // A nonstandard CSP header that was introduced for CSP v.1
  // https://www.w3.org/TR/2011/WD-CSP-20111129/ and used by the Firefox until
  // version 23 and the Internet Explorer version 10.
  // Please, use CONTENT_SECURITY_POLICY to pass the CSP.
  static const char X_CONTENT_SECURITY_POLICY[];
  // A nonstandard CSP header that was introduced for CSP v.1
  // https://www.w3.org/TR/2011/WD-CSP-20111129/ and used by the Firefox until
  // version 23 and Internet Explorer version 10.
  // Please, use CONTENT_SECURITY_POLICY_REPORT_ONLY to pass the CSP.
  static const char X_CONTENT_SECURITY_POLICY_REPORT_ONLY[];
  // A nonstandard CSP header that was introduced for CSP v.1
  // https://www.w3.org/TR/2011/WD-CSP-20111129/ and used by the Chrome until
  // version 25. Please, use CONTENT_SECURITY_POLICY to pass the CSP.
  static const char X_WEBKIT_CSP[];
  // A nonstandard CSP header that was introduced for CSP v.1
  // https://www.w3.org/TR/2011/WD-CSP-20111129/ and used by the Chrome until
  // version 25.
  // Please, use CONTENT_SECURITY_POLICY_REPORT_ONLY to pass the CSP.
  static const char X_WEBKIT_CSP_REPORT_ONLY[];
  static const char CONTENT_TYPE[];
  static const char CONTENT_MD5[];
  // A header, introduced by Microsoft, to modify how browsers
  // interpret Content-Type:
  // http://blogs.msdn.com/ie/archive/2008/09/02/ie8-security-part-vi-beta-2-update.aspx
  static const char X_CONTENT_TYPE_OPTIONS[];
  static const char COOKIE[];
  static const char COOKIE2[];
  static const char DATE[];
  static const char DAV[];
  static const char DEPTH[];
  static const char DESTINATION[];
  static const char DNT[];
  // https://tools.ietf.org/html/rfc8470
  static const char EARLY_DATA[];
  static const char ETAG[];
  static const char EXPECT[];
  static const char EXPIRES[];
  static const char FOLLOW_ONLY_WHEN_PRERENDER_SHOWN[];
  // Supersedes X-Forwarded-For (https://tools.ietf.org/html/rfc7239).
  static const char FORWARDED[];
  static const char FROM[];
  static const char HOST[];
  // http://httpwg.org/specs/rfc7540.html#Http2SettingsHeader
  static const char HTTP2_SETTINGS[];
  static const char IF[];
  static const char IF_MATCH[];
  static const char IF_MODIFIED_SINCE[];
  static const char IF_NONE_MATCH[];
  static const char IF_RANGE[];
  static const char IF_UNMODIFIED_SINCE[];
  static const char KEEP_ALIVE[];
  static const char LABEL[];
  static const char LAST_MODIFIED[];
  static const char LINK[];
  static const char LOCATION[];
  static const char LOCK_TOKEN[];
  static const char MAX_FORWARDS[];
  static const char MS_AUTHOR_VIA[];
  static const char ORIGIN[];
  static const char OVERWRITE_HDR[];
  static const char P3P[];
  // http://html.spec.whatwg.org/multipage/semantics.html#hyperlink-auditing
  static const char PING_FROM[];
  // http://html.spec.whatwg.org/multipage/semantics.html#hyperlink-auditing
  static const char PING_TO[];
  static const char PRAGMA[];
  static const char PROXY_CONNECTION[];
  static const char PROXY_AUTHENTICATE[];
  static const char PROXY_AUTHORIZATION[];
  // http://tools.ietf.org/html/draft-ietf-websec-key-pinning
  static const char PUBLIC_KEY_PINS[];
  static const char PUBLIC_KEY_PINS_REPORT_ONLY[];
  static const char RANGE[];
  static const char REFERER[];
  // https://www.w3.org/TR/referrer-policy/
  static const char REFERRER_POLICY[];
  static const char REFERRER_POLICY_NO_REFERRER[];
  static const char REFERRER_POLICY_NO_REFFERER_WHEN_DOWNGRADE[];
  static const char REFERRER_POLICY_SAME_ORIGIN[];
  static const char REFERRER_POLICY_ORIGIN[];
  static const char REFERRER_POLICY_STRICT_ORIGIN[];
  static const char REFERRER_POLICY_ORIGIN_WHEN_CROSS_ORIGIN[];
  static const char REFERRER_POLICY_STRICT_ORIGIN_WHEN_CROSS_ORIGIN[];
  static const char REFERRER_POLICY_UNSAFE_URL[];
  static const char REFRESH[];
  static const char RETRY_AFTER[];
  // https://github.com/mikewest/sec-metadata
  static const char SEC_METADATA[];
  // https://tools.ietf.org/html/draft-ietf-tokbind-https
  static const char SEC_TOKEN_BINDING[];
  // https://tools.ietf.org/html/draft-ietf-tokbind-ttrp
  static const char SEC_PROVIDED_TOKEN_BINDING_ID[];
  static const char SEC_REFERRED_TOKEN_BINDING_ID[];
  static const char SERVER[];
  // https://www.w3.org/TR/server-timing/
  static const char SERVER_TIMING[];
  // https://www.w3.org/TR/service-workers/#update-algorithm
  static const char SERVICE_WORKER[];
  static const char SERVICE_WORKER_ALLOWED[];
  // https://developers.google.com/web/updates/2017/02/navigation-preload
  static const char SERVICE_WORKER_NAVIGATION_PRELOAD[];
  static const char SET_COOKIE[];
  static const char SET_COOKIE2[];
  static const char STATUS_URI[];
  // HSTS http://tools.ietf.org/html/draft-ietf-websec-strict-transport-sec
  static const char STRICT_TRANSPORT_SECURITY[];
  static const char TIMEOUT[];
  // http://www.w3.org/TR/2011/WD-resource-timing-20110524/#cross-origin-resources
  static const char TIMING_ALLOW_ORIGIN[];
  // http://www.w3.org/2011/tracking-protection/drafts/tracking-dnt.html#response-header-field
  static const char TK[];
  static const char TRAILER[];
  static const char TRAILERS[];
  static const char TRANSFER_ENCODING[];
  static const char TRANSFER_ENCODING_ABBRV[];
  static const char UPGRADE[];
  static const char USER_AGENT[];
  static const char VARY[];
  static const char VIA[];
  static const char WARNING[];
  static const char WWW_AUTHENTICATE[];
};

}  // namespace net_http
}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_UTIL_NET_HTTP_PUBLIC_HEADER_NAMES_H_
