# Description: Tensorflow Serving storage path sources.

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
        ],
    ),
)

cc_library(
    name = "static_storage_path_source",
    srcs = ["static_storage_path_source.cc"],
    hdrs = ["static_storage_path_source.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":static_storage_path_source_cc_proto",
        "//tensorflow_serving/core:servable_data",
        "//tensorflow_serving/core:servable_id",
        "//tensorflow_serving/core:source",
        "//tensorflow_serving/core:storage_path",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

serving_proto_library(
    name = "static_storage_path_source_proto",
    srcs = ["static_storage_path_source.proto"],
    cc_api_version = 2,
    visibility = ["//visibility:public"],
)

cc_test(
    name = "static_storage_path_source_test",
    size = "small",
    srcs = ["static_storage_path_source_test.cc"],
    deps = [
        ":static_storage_path_source",
        "//tensorflow_serving/core:servable_data",
        "//tensorflow_serving/core:target",
        "//tensorflow_serving/core/test_util:mock_storage_path_target",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/test_util",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "file_system_storage_path_source",
    srcs = ["file_system_storage_path_source.cc"],
    hdrs = ["file_system_storage_path_source.h"],
    visibility = ["//visibility:public"],
    deps =
        [
            "//tensorflow_serving/config:file_system_storage_path_source_cc_proto",
            "//tensorflow_serving/core:servable_data",
            "//tensorflow_serving/core:servable_id",
            "//tensorflow_serving/core:source",
            "//tensorflow_serving/core:storage_path",
            "@com_google_absl//absl/types:variant",
            "@org_tensorflow//tensorflow/core:lib",
            "@org_tensorflow//tensorflow/core:tensorflow",
            "@org_tensorflow//tensorflow/core/kernels/batching_util:periodic_function_dynamic",
        ],
)

cc_test(
    name = "file_system_storage_path_source_test",
    srcs =
        [
            "file_system_storage_path_source_test.cc",
        ],
    deps = [
        ":file_system_storage_path_source",
        "//tensorflow_serving/config:file_system_storage_path_source_cc_proto",
        "//tensorflow_serving/core:servable_data",
        "//tensorflow_serving/core:target",
        "//tensorflow_serving/core/test_util:mock_storage_path_target",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/test_util",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)
