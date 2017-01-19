# Description: Tensorflow Serving configs.

package(
    default_visibility = ["//visibility:public"],
    features = ["-layering_check"],
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

load("//tensorflow_serving:serving.bzl", "serving_proto_library")

serving_proto_library(
    name = "model_server_config_proto",
    srcs = ["model_server_config.proto"],
    cc_api_version = 2,
    deps = [
        ":logging_config_proto",
        "//tensorflow_serving/sources/storage_path:file_system_storage_path_source_proto",
        "@protobuf//:cc_wkt_protos",
    ],
)

serving_proto_library(
    name = "platform_config_proto",
    srcs = ["platform_config.proto"],
    cc_api_version = 2,
    deps = [
        "@protobuf//:cc_wkt_protos",
    ],
)

serving_proto_library(
    name = "log_collector_config_proto",
    srcs = ["log_collector_config.proto"],
    cc_api_version = 2,
    deps = [
    ],
)

serving_proto_library(
    name = "logging_config_proto",
    srcs = ["logging_config.proto"],
    cc_api_version = 2,
    deps = [
        ":log_collector_config_proto",
    ],
)
