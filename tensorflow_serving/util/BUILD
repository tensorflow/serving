# Description: Tensorflow Serving utils.

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

###############################################################################
#                  Public targets
###############################################################################

cc_library(
    name = "event_bus",
    hdrs = ["event_bus.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:tensorflow",
    ],
)

cc_library(
    name = "any_ptr",
    hdrs = ["any_ptr.h"],
    visibility = ["//visibility:public"],
)

###############################################################################
#                  Internal targets
###############################################################################

cc_library(
    name = "observer",
    hdrs = ["observer.h"],
    deps = [
        "@tf//tensorflow/core:tensorflow",
    ],
)

cc_test(
    name = "observer_test",
    size = "small",
    srcs = ["observer_test.cc"],
    deps = [
        "@tf//tensorflow/core:lib",
        ":observer",
        ":periodic_function",
        "//tensorflow_serving/core/test_util:test_main",
    ],
)

cc_test(
    name = "event_bus_test",
    size = "small",
    srcs = ["event_bus_test.cc"],
    deps = [
        ":event_bus",
        "//tensorflow_serving/core/test_util:test_main",
    ],
)

cc_library(
    name = "fast_read_dynamic_ptr",
    hdrs = ["fast_read_dynamic_ptr.h"],
    deps = [
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:tensorflow",
    ],
)

cc_test(
    name = "fast_read_dynamic_ptr_test",
    srcs = ["fast_read_dynamic_ptr_test.cc"],
    deps = [
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:tensorflow",
        ":fast_read_dynamic_ptr",
        "//tensorflow_serving/core/test_util:test_main",
    ],
)

cc_test(
    name = "fast_read_dynamic_ptr_benchmark",
    srcs = ["fast_read_dynamic_ptr_benchmark.cc"],
    deps = [
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:tensorflow",
        "@tf//tensorflow/core:test",
        ":fast_read_dynamic_ptr",
        ":periodic_function",
    ],
)

cc_library(
    name = "optional",
    srcs = ["optional.cc"],
    hdrs = ["optional.h"],
    deps = [
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:tensorflow",
    ],
)

cc_test(
    name = "optional_test",
    size = "small",
    srcs = [
        "optional_test.cc",
    ],
    deps = [
        "@tf//tensorflow/core:lib",
        ":optional",
        "//tensorflow_serving/core/test_util:test_main",
    ],
)

cc_library(
    name = "periodic_function",
    srcs = ["periodic_function.cc"],
    hdrs = ["periodic_function.h"],
    deps = [
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:tensorflow",
    ],
)

cc_test(
    name = "periodic_function_test",
    srcs = ["periodic_function_test.cc"],
    deps = [
        "@tf//tensorflow/core:tensorflow",
        ":periodic_function",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/test_util:fake_clock_env",
    ],
)

cc_test(
    name = "any_ptr_test",
    srcs = [
        "any_ptr_test.cc",
    ],
    deps = [
        ":any_ptr",
        "//tensorflow_serving/core/test_util:test_main",
    ],
)

cc_library(
    name = "cleanup",
    hdrs = ["cleanup.h"],
    deps = [
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:tensorflow",
    ],
)

cc_test(
    name = "cleanup_test",
    size = "small",
    srcs = ["cleanup_test.cc"],
    deps = [
        ":cleanup",
        "//tensorflow_serving/core/test_util:test_main",
    ],
)

cc_library(
    name = "executor",
    hdrs = ["executor.h"],
    deps = [
    ],
)

cc_library(
    name = "inline_executor",
    srcs = ["inline_executor.cc"],
    hdrs = ["inline_executor.h"],
    deps = [
        "@tf//tensorflow/core:lib",
        ":executor",
    ],
)

cc_test(
    name = "inline_executor_test",
    srcs = ["inline_executor_test.cc"],
    deps = [
        ":inline_executor",
        "//tensorflow_serving/core/test_util:test_main",
    ],
)

cc_library(
    name = "threadpool_executor",
    srcs = ["threadpool_executor.cc"],
    hdrs = ["threadpool_executor.h"],
    deps = [
        "@tf//tensorflow/core:lib",
        ":executor",
    ],
)

cc_test(
    name = "threadpool_executor_test",
    srcs = ["threadpool_executor_test.cc"],
    deps = [
        ":threadpool_executor",
        "//tensorflow_serving/core/test_util:test_main",
    ],
)

cc_library(
    name = "unique_ptr_with_deps",
    hdrs = ["unique_ptr_with_deps.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@tf//tensorflow/core:lib",
        ":any_ptr",
    ],
)

cc_test(
    name = "unique_ptr_with_deps_test",
    size = "small",
    srcs = [
        "unique_ptr_with_deps_test.cc",
    ],
    deps = [
        ":unique_ptr_with_deps",
        "//tensorflow_serving/core/test_util:test_main",
    ],
)
