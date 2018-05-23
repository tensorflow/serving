# Description: Tensorflow Serving batching.

package(
    default_visibility = ["//tensorflow_serving:internal"],
    features = ["-layering_check"],
)

licenses(["notice"])  # Apache 2.0

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
    name = "streaming_batch_scheduler",
    srcs = ["streaming_batch_scheduler.cc"],
    hdrs = ["streaming_batch_scheduler.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":batch_scheduler_retrier",
        "//tensorflow_serving/util:optional",
        "@org_tensorflow//tensorflow/contrib/batching:batch_scheduler",
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
        "@org_tensorflow//tensorflow/contrib/batching/test_util:fake_clock_env",
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
        "//tensorflow_serving/batching:batching_util",
        "//tensorflow_serving/servables/tensorflow:serving_session",
        "//tensorflow_serving/util:cleanup",
        "//tensorflow_serving/util:hash",
        "//tensorflow_serving/util:optional",
        "@org_tensorflow//tensorflow/contrib/batching:basic_batch_scheduler",
        "@org_tensorflow//tensorflow/contrib/batching:batch_scheduler",
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
    data = [
        "//tensorflow_serving/batching/testdata:matrix_half_plus_two",
        "@org_tensorflow//tensorflow/cc/saved_model:saved_model_half_plus_two",
    ],
    deps = [
        ":batching_session",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/servables/tensorflow:serving_session",
        "//tensorflow_serving/test_util",
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
        "@org_tensorflow//tensorflow/cc/saved_model:tag_constants",
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
        "@org_tensorflow//tensorflow/contrib/batching:batch_scheduler",
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
        "@org_tensorflow//tensorflow/contrib/batching/test_util:fake_clock_env",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "batching_util",
    srcs = ["batching_util.cc"],
    hdrs = ["batching_util.h"],
    deps = [
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "batching_util_test",
    srcs = [
        "batching_util_test.cc",
    ],
    deps = [
        ":batching_util",
        "//tensorflow_serving/core/test_util:test_main",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)
