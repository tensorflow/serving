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
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:tensorflow",
    ],
)

cc_test(
    name = "batch_scheduler_test",
    srcs = [
        "batch_scheduler_test.cc",
    ],
    deps = [
        ":batch_scheduler",
        "//tensorflow_serving/core/test_util:test_main",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "shared_batch_scheduler",
    hdrs = ["shared_batch_scheduler.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":batch_scheduler",
        "//tensorflow_serving/util:periodic_function",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "shared_batch_scheduler_test",
    srcs = [
        "shared_batch_scheduler_test.cc",
    ],
    deps = [
        ":shared_batch_scheduler",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/test_util:fake_clock_env",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "basic_batch_scheduler",
    hdrs = ["basic_batch_scheduler.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":shared_batch_scheduler",
    ],
)

cc_test(
    name = "basic_batch_scheduler_test",
    srcs = [
        "basic_batch_scheduler_test.cc",
    ],
    deps = [
        ":basic_batch_scheduler",
        ":batch_scheduler",
        "//tensorflow_serving/core/test_util:test_main",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
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
        ":basic_batch_scheduler",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:tensorflow",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "streaming_batch_scheduler",
    srcs = ["streaming_batch_scheduler.cc"],
    hdrs = ["streaming_batch_scheduler.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":batch_scheduler",
        ":batch_scheduler_retrier",
        "//tensorflow_serving/util:optional",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "streaming_batch_scheduler_test",
    srcs = [
        "streaming_batch_scheduler_test.cc",
    ],
    deps = [
        ":streaming_batch_scheduler",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/test_util:fake_clock_env",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:test",
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
        ":basic_batch_scheduler",
        ":batch_scheduler",
        "//tensorflow_serving/servables/tensorflow:serving_session",
        "//tensorflow_serving/util:cleanup",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:tensorflow",
    ],
)

cc_test(
    name = "batching_session_test",
    srcs = [
        "batching_session_test.cc",
    ],
    data = ["@org_tensorflow//tensorflow/contrib/session_bundle/example:half_plus_two"],
    deps = [
        ":batching_session",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/servables/tensorflow:serving_session",
        "//tensorflow_serving/test_util",
        "@org_tensorflow//tensorflow/contrib/session_bundle",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
        "@org_tensorflow//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "batch_scheduler_retrier",
    hdrs = ["batch_scheduler_retrier.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":batch_scheduler",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "batch_scheduler_retrier_test",
    srcs = [
        "batch_scheduler_retrier_test.cc",
    ],
    deps = [
        ":batch_scheduler_retrier",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/test_util:fake_clock_env",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:test",
    ],
)
