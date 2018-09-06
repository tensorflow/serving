# Description: net_http/client/testing

package(
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])  # Apache 2.0

cc_binary(
    name = "evhttp_echo_client",
    srcs = ["evhttp_echo_client.cc"],
    deps = [
        "//tensorflow_serving/util/net_http/client:evhttp_client",
    ],
)
