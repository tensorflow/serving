# Description: implementations of HTTP Server

package(
    default_visibility = [
        "//tensorflow_serving/util/net_http:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "evhttp_server",
    srcs = [
        "evhttp_request.cc",
        "evhttp_server.cc",
    ],
    hdrs = [
        "evhttp_request.h",
        "evhttp_server.h",
        "server_support.h",
    ],
    deps = [
        "//tensorflow_serving/util/net_http/compression:gzip_zlib",
        "//tensorflow_serving/util/net_http/internal:net_logging",
        "//tensorflow_serving/util/net_http/server/public:http_server_api",
        "@com_github_libevent_libevent//:libevent",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@zlib_archive//:zlib",
    ],
)

cc_test(
    name = "evhttp_server_test",
    size = "medium",
    srcs = ["evhttp_server_test.cc"],
    features = ["-layering_check"],
    deps = [
        ":evhttp_server",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/util/net_http/client:evhttp_client",
        "//tensorflow_serving/util/net_http/internal:fixed_thread_pool",
        "//tensorflow_serving/util/net_http/server/public:http_server",
        "//tensorflow_serving/util/net_http/server/public:http_server_api",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_test(
    name = "evhttp_request_test",
    size = "medium",
    srcs = ["evhttp_request_test.cc"],
    features = ["-layering_check"],
    deps = [
        ":evhttp_server",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/util/net_http/client:evhttp_client",
        "//tensorflow_serving/util/net_http/compression:gzip_zlib",
        "//tensorflow_serving/util/net_http/internal:fixed_thread_pool",
        "//tensorflow_serving/util/net_http/server/public:http_server",
        "//tensorflow_serving/util/net_http/server/public:http_server_api",
        "@com_google_absl//absl/memory",
    ],
)
