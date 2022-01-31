# Description: net_http/client/test_client/testing

package(
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])

cc_binary(
    name = "evhttp_echo_client",
    srcs = ["evhttp_echo_client.cc"],
    deps = [
        "//tensorflow_serving/util/net_http/client/test_client/internal:evhttp_client",
    ],
)
