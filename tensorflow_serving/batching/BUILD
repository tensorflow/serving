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
    name = "shared_batch_scheduler",
    hdrs = ["shared_batch_scheduler.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@tf//tensorflow/core:lib",
        ":batch_scheduler",
        "//tensorflow_serving/util:periodic_function",
    ],
)

cc_test(
    name = "shared_batch_scheduler_test",
    srcs = [
        "shared_batch_scheduler_test.cc",
    ],
    deps = [
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:protos_all_cc",
        "@tf//tensorflow/core:test",
        ":shared_batch_scheduler",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/test_util:fake_clock_env",
    ],
)

cc_library(
    name = "basic_batch_scheduler",
    hdrs = ["basic_batch_scheduler.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":batch_scheduler_retrier",
        ":shared_batch_scheduler",
    ],
)

cc_test(
    name = "basic_batch_scheduler_test",
    srcs = [
        "basic_batch_scheduler_test.cc",
    ],
    deps = [
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:test",
        ":basic_batch_scheduler",
        ":batch_scheduler",
        "//tensorflow_serving/core/test_util:test_main",
    ],
)

cc_test(
    name = "basic_batch_scheduler_benchmark",
    srcs = ["basic_batch_scheduler_benchmark.cc"],
    tags = [
        "local",
        "manual",
    ],
    deps = [
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:tensorflow",
        "@tf//tensorflow/core:test",
        ":basic_batch_scheduler",
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
        ":batch_scheduler_retrier",
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
    name = "batching_session",
    srcs = ["batching_session.cc"],
    hdrs = ["batching_session.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@tf//tensorflow/core:core_cpu",
        "@tf//tensorflow/core:framework",
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:tensorflow",
        ":basic_batch_scheduler",
        ":batch_scheduler",
        ":batch_scheduler_retrier",
        "//tensorflow_serving/servables/tensorflow:serving_session",
        "//tensorflow_serving/util:cleanup",
    ],
)

cc_test(
    name = "batching_session_test",
    srcs = [
        "batching_session_test.cc",
    ],
    data = ["//tensorflow_serving/session_bundle/example:half_plus_two"],
    deps = [
        "@tf//tensorflow/core:core_cpu",
        "@tf//tensorflow/core:framework",
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:test",
        "@tf//tensorflow/core:testlib",
        ":batch_scheduler_retrier",
        ":batching_session",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/servables/tensorflow:serving_session",
        "//tensorflow_serving/session_bundle",
        "//tensorflow_serving/test_util",
    ],
)

cc_library(
    name = "batch_scheduler_retrier",
    hdrs = ["batch_scheduler_retrier.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@tf//tensorflow/core:lib",
        ":batch_scheduler",
    ],
)

cc_test(
    name = "batch_scheduler_retrier_test",
    srcs = [
        "batch_scheduler_retrier_test.cc",
    ],
    deps = [
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:protos_all_cc",
        "@tf//tensorflow/core:test",
        ":batch_scheduler_retrier",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/test_util:fake_clock_env",
    ],
)
