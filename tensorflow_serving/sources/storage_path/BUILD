# Description: Tensorflow Serving storage path sources.

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

cc_library(
    name = "static_storage_path_source",
    srcs = ["static_storage_path_source.cc"],
    hdrs = ["static_storage_path_source.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@tf//tensorflow/core:lib",
        ":static_storage_path_source_proto",
        "//tensorflow_serving/core:servable_data",
        "//tensorflow_serving/core:servable_id",
        "//tensorflow_serving/core:source",
        "//tensorflow_serving/core:storage_path",
    ],
)

load("//tensorflow_serving:serving.bzl", "serving_proto_library")

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
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:test",
        ":static_storage_path_source",
        "//tensorflow_serving/core:servable_data",
        "//tensorflow_serving/core:target",
        "//tensorflow_serving/core/test_util:mock_storage_path_target",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/test_util",
    ],
)

cc_library(
    name = "file_system_storage_path_source",
    srcs = ["file_system_storage_path_source.cc"],
    hdrs = ["file_system_storage_path_source.h"],
    visibility = ["//visibility:public"],
    deps =
        [
            "@tf//tensorflow/core:lib",
            "@tf//tensorflow/core:tensorflow",
            ":file_system_storage_path_source_proto",
            "//tensorflow_serving/core:servable_data",
            "//tensorflow_serving/core:servable_id",
            "//tensorflow_serving/core:source",
            "//tensorflow_serving/core:storage_path",
            "//tensorflow_serving/util:periodic_function",
        ],
)

serving_proto_library(
    name = "file_system_storage_path_source_proto",
    srcs = ["file_system_storage_path_source.proto"],
    cc_api_version = 2,
    visibility = ["//visibility:public"],
)

cc_test(
    name = "file_system_storage_path_source_test",
    srcs =
        [
            "file_system_storage_path_source_test.cc",
        ],
    deps = [
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:test",
        ":file_system_storage_path_source",
        ":file_system_storage_path_source_proto",
        "//tensorflow_serving/core:servable_data",
        "//tensorflow_serving/core:target",
        "//tensorflow_serving/core/test_util:mock_storage_path_target",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/test_util",
    ],
)
