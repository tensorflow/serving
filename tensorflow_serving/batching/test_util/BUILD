# Description: Tensorflow Serving batching test utilities.

package(
    default_visibility = ["//tensorflow_serving:internal"],
    features = [
        "-layering_check",
        "-parse_headers",
    ],
)

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
    name = "puppet_batch_scheduler",
    testonly = 1,
    hdrs = ["puppet_batch_scheduler.h"],
    visibility = ["//visibility:private"],
    deps = [
        "@tf//tensorflow/core:tensorflow",
        "//tensorflow_serving/batching:batch_scheduler",
    ],
)

cc_test(
    name = "puppet_batch_scheduler_test",
    srcs = [
        "puppet_batch_scheduler_test.cc",
    ],
    deps = [
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:test",
        ":puppet_batch_scheduler",
        "//tensorflow_serving/core/test_util:test_main",
    ],
)
