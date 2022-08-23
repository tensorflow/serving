# Description: net_http/server/testing

package(
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])

cc_binary(
    name = "evhttp_echo_server",
    srcs = ["evhttp_echo_server.cc"],
    deps = [
        "//tensorflow_serving/util/net_http/server/public:http_server",
        "//tensorflow_serving/util/net_http/server/public:http_server_api",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)
