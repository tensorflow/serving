# Description: net_http/socket testing utils

package(
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])  # Apache 2.0

cc_binary(
    name = "ev_print_req_server",
    srcs = ["ev_print_req_server.cc"],
    deps = [
        "@com_github_libevent_libevent//:libevent",
        "@com_google_absl//absl/strings",
    ],
)

cc_binary(
    name = "ev_fetch_client",
    srcs = ["ev_fetch_client.cc"],
    deps = [
        "@com_github_libevent_libevent//:libevent",
    ],
)
