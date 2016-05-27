# Description: Tensorflow Serving test utils.

package(default_visibility = [
    "//tensorflow_serving:internal",
])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
)

cc_library(
    name = "test_util",
    testonly = 1,
    srcs = ["test_util.cc"],
    hdrs = ["test_util.h"],
    deps = [
        "@tf//google/protobuf:protobuf",
        "@tf//tensorflow/core",
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:test",
        "//external:gtest",
    ],
)

cc_library(
    name = "fake_clock_env",
    testonly = 1,
    srcs = ["fake_clock_env.cc"],
    hdrs = ["fake_clock_env.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:tensorflow",
    ],
)

py_library(
    name = "test_util_py",
    testonly = 1,
    srcs = ["test_util.py"],
    deps = [
        "@tf//tensorflow:tensorflow_py",
    ],
)
