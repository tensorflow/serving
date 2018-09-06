# Description: shared code for net_http

package(
    default_visibility = [
        "//tensorflow_serving/util/net_http:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "fixed_thread_pool",
    testonly = 1,
    hdrs = ["fixed_thread_pool.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
    ],
)
