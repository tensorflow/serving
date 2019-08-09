# Description: a lightweight http client

package(
    default_visibility = [
        "//tensorflow_serving/util/net_http:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "evhttp_client",
    srcs = [
        "evhttp_connection.cc",
    ],
    hdrs = [
        "evhttp_connection.h",
    ],
    deps = [
        "//tensorflow_serving/util/net_http/internal:net_logging",
        "//tensorflow_serving/util/net_http/server/public:http_server_api",
        "@com_github_libevent_libevent//:libevent",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)
