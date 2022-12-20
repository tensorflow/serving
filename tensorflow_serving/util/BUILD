# Description: Tensorflow Serving utils.

load("//tensorflow_serving:serving.bzl", "serving_proto_library")

package(
    default_visibility = [
        "//tensorflow_serving:internal",
    ],
    features = ["-layering_check"],
)

licenses(["notice"])

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
            "**/google_*",
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
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "any_ptr",
    hdrs = ["any_ptr.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "class_registration",
    hdrs = [
        "class_registration.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":class_registration_util",
        "@com_google_protobuf//:cc_wkt_protos",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "observer",
    hdrs = ["observer.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "prometheus_exporter",
    srcs = ["prometheus_exporter.cc"],
    hdrs = ["prometheus_exporter.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_googlesource_code_re2//:re2",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

###############################################################################
#                  Internal targets
###############################################################################
cc_library(
    name = "hash",
    srcs = ["hash.cc"],
    hdrs = ["hash.h"],
    deps = [
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "observer_test",
    size = "small",
    srcs = ["observer_test.cc"],
    deps = [
        ":observer",
        "//tensorflow_serving/core/test_util:test_main",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core/kernels/batching_util:periodic_function",
    ],
)

cc_test(
    name = "prometheus_exporter_test",
    size = "small",
    srcs = ["prometheus_exporter_test.cc"],
    deps = [
        ":prometheus_exporter",
        "//tensorflow_serving/core/test_util:test_main",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "event_bus_test",
    size = "small",
    srcs = ["event_bus_test.cc"],
    deps = [
        ":event_bus",
        "//tensorflow_serving/core/test_util:test_main",
        "@org_tensorflow//tensorflow/core/kernels/batching_util:fake_clock_env",
    ],
)

cc_library(
    name = "fast_read_dynamic_ptr",
    hdrs = ["fast_read_dynamic_ptr.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "fast_read_dynamic_ptr_test",
    srcs = ["fast_read_dynamic_ptr_test.cc"],
    deps = [
        ":fast_read_dynamic_ptr",
        "//tensorflow_serving/core/test_util:test_main",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:tensorflow",
    ],
)

cc_test(
    name = "fast_read_dynamic_ptr_benchmark",
    timeout = "long",
    srcs = ["fast_read_dynamic_ptr_benchmark.cc"],
    deps = [
        ":fast_read_dynamic_ptr",
        "@com_google_absl//absl/time",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:tensorflow",
        "@org_tensorflow//tensorflow/core:test",
        "@org_tensorflow//tensorflow/core/kernels/batching_util:periodic_function_dynamic",
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
        "@com_google_absl//absl/base:core_headers",
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
        ":executor",
        "@org_tensorflow//tensorflow/core:lib",
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
        ":executor",
        "@org_tensorflow//tensorflow/core:lib",
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
        ":any_ptr",
        "@org_tensorflow//tensorflow/core:lib",
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

cc_library(
    name = "file_probing_env",
    srcs = ["file_probing_env.cc"],
    hdrs = ["file_probing_env.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "class_registration_util",
    srcs = ["class_registration_util.cc"],
    hdrs = ["class_registration_util.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "class_registration_test",
    srcs = [
        "class_registration_test.cc",
    ],
    deps = [
        ":class_registration",
        ":class_registration_test_cc_proto",
        "//tensorflow_serving/core/test_util:test_main",
    ],
)

cc_library(
    name = "retrier",
    srcs = ["retrier.cc"],
    hdrs = ["retrier.h"],
    deps = [
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "retrier_test",
    srcs = ["retrier_test.cc"],
    deps = [
        ":retrier",
        "//tensorflow_serving/core/test_util:test_main",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "status_util",
    srcs = ["status_util.cc"],
    hdrs = ["status_util.h"],
    deps = [
        "//tensorflow_serving/apis:status_cc_proto",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "json_tensor",
    srcs = ["json_tensor.cc"],
    hdrs = ["json_tensor.h"],
    deps = [
        "//tensorflow_serving/apis:classification_cc_proto",
        "//tensorflow_serving/apis:input_cc_proto",
        "//tensorflow_serving/apis:model_cc_proto",
        "//tensorflow_serving/apis:predict_cc_proto",
        "//tensorflow_serving/apis:regression_cc_proto",
        "@com_github_tencent_rapidjson//:rapidjson",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_test(
    name = "status_util_test",
    srcs = ["status_util_test.cc"],
    deps = [
        ":status_util",
        "//tensorflow_serving/core/test_util:test_main",
    ],
)

cc_test(
    name = "json_tensor_test",
    srcs = ["json_tensor_test.cc"],
    deps = [
        ":json_tensor",
        "//tensorflow_serving/apis:classification_cc_proto",
        "//tensorflow_serving/apis:model_cc_proto",
        "//tensorflow_serving/apis:predict_cc_proto",
        "//tensorflow_serving/apis:regression_cc_proto",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/test_util",
        "@com_google_protobuf//:protobuf",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "proto_util",
    srcs = ["proto_util.h"],
    hdrs = ["proto_util.h"],
    deps = ["@org_tensorflow//tensorflow/core:lib"],
)

cc_test(
    name = "proto_util_test",
    srcs = ["proto_util_test.cc"],
    deps = [
        ":proto_util",
        "//tensorflow_serving/apis:status_cc_proto",
        "//tensorflow_serving/core/test_util:test_main",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

serving_proto_library(
    name = "class_registration_test_proto",
    testonly = 1,
    srcs = ["class_registration_test.proto"],
    cc_api_version = 2,
    deps = [
        "@com_google_protobuf//:cc_wkt_protos",
    ],
)

cc_library(
    name = "oss_or_google",
    hdrs = ["oss_or_google.h"],
)
