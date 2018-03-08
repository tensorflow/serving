# Description: Tensorflow Serving configs.

package(
    default_visibility = ["//visibility:public"],
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

load("//tensorflow_serving:serving.bzl", "serving_proto_library")
load("//tensorflow_serving:serving.bzl", "serving_proto_library_py")

serving_proto_library(
    name = "model_server_config_proto",
    srcs = ["model_server_config.proto"],
    cc_api_version = 2,
    deps = [
        ":logging_config_proto",
        "//tensorflow_serving/sources/storage_path:file_system_storage_path_source_proto",
        "@protobuf_archive//:cc_wkt_protos",
    ],
)

serving_proto_library_py(
    name = "model_server_config_proto_py_pb2",
    srcs = ["model_server_config.proto"],
    proto_library = "model_server_config_proto",
)

serving_proto_library(
    name = "platform_config_proto",
    srcs = ["platform_config.proto"],
    cc_api_version = 2,
    deps = [
        "@protobuf_archive//:cc_wkt_protos",
    ],
)

serving_proto_library(
    name = "log_collector_config_proto",
    srcs = ["log_collector_config.proto"],
    cc_api_version = 2,
    java_api_version = 2,
    deps = [
    ],
)

serving_proto_library(
    name = "logging_config_proto",
    srcs = ["logging_config.proto"],
    cc_api_version = 2,
    java_api_version = 2,
    deps = [
        ":log_collector_config_proto",
    ],
)
