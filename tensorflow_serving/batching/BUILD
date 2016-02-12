# Description: Tensorflow Serving batching.

package(
    default_visibility = ["//tensorflow_serving:internal"],
    features = [
        "-layering_check",
        "-parse_headers",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

cc_library(
    name = "batch_scheduler",
    hdrs = ["batch_scheduler.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:tensorflow",
    ],
)

cc_test(
    name = "batch_scheduler_test",
    srcs = [
        "batch_scheduler_test.cc",
    ],
    deps = [
        "@tf//tensorflow/core:lib",
        ":batch_scheduler",
        "//tensorflow_serving/core/test_util:test_main",
    ],
)

cc_library(
    name = "streaming_batch_scheduler",
    srcs = ["streaming_batch_scheduler.cc"],
    hdrs = ["streaming_batch_scheduler.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@tf//tensorflow/core:lib",
        ":batch_scheduler",
        ":retrier",
        "//tensorflow_serving/util:optional",
    ],
)

cc_test(
    name = "streaming_batch_scheduler_test",
    srcs = [
        "streaming_batch_scheduler_test.cc",
    ],
    deps = [
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:protos_all_cc",
        "@tf//tensorflow/core:test",
        ":streaming_batch_scheduler",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/test_util:fake_clock_env",
    ],
)

cc_library(
    name = "retrier",
    hdrs = ["retrier.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@tf//tensorflow/core:lib",
        ":batch_scheduler",
    ],
)

cc_test(
    name = "retrier_test",
    srcs = [
        "retrier_test.cc",
    ],
    deps = [
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:protos_all_cc",
        "@tf//tensorflow/core:test",
        ":retrier",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/test_util:fake_clock_env",
    ],
)
