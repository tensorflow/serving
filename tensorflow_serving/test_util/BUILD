# Description: Tensorflow Serving test utils.

package(default_visibility = [
    "//tensorflow_serving:internal",
])

licenses(["notice"])

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
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@com_google_googletest//:gtest",
        "@com_google_protobuf//:protobuf",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:session_options",
        "@org_tensorflow//tensorflow/core:test",
    ],
)
