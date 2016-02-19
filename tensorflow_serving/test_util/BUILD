# Description: Tensorflow Serving test utils.

package(
    default_visibility = [
        "//tensorflow_serving:internal",
    ],
    features = [
        "-layering_check",
        "-parse_headers",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

cc_library(
    name = "test_util",
    testonly = 1,
    hdrs = ["test_util.h"],
    visibility = [
        "//tensorflow_serving:internal",
    ],
    deps = [
        "@tf//google/protobuf:protobuf",
        "@tf//tensorflow/core",
        "@tf//tensorflow/core:lib",
    ],
)

cc_library(
    name = "fake_clock_env",
    testonly = 1,
    srcs = ["fake_clock_env.cc"],
    hdrs = ["fake_clock_env.h"],
    visibility = [
        "//tensorflow_serving:internal",
    ],
    deps = [
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:tensorflow",
    ],
)
