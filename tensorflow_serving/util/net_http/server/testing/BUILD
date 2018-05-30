# Description: net_http/server/testing

package(
    default_visibility = [
        "//tensorflow_serving:internal",
    ],
    features = ["-layering_check"],
)

licenses(["notice"])  # Apache 2.0

cc_binary(
    name = "evhttp_echo_server",
    srcs = ["evhttp_echo_server.cc"],
    deps = [
        "//tensorflow_serving/util/net_http/server/public:http_server",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)
