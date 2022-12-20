# Description: Tensorflow Serving configs.

package(
    default_visibility = ["//visibility:public"],
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

load("//tensorflow_serving:serving.bzl", "serving_proto_library")
load("//tensorflow_serving:serving.bzl", "serving_proto_library_py")

serving_proto_library(
    name = "file_system_storage_path_source_proto",
    srcs = ["file_system_storage_path_source.proto"],
    cc_api_version = 2,
)

serving_proto_library_py(
    name = "file_system_storage_path_source_proto_py_pb2",
    srcs = ["file_system_storage_path_source.proto"],
    proto_library = "file_system_storage_path_source_proto",
)

serving_proto_library(
    name = "model_server_config_proto",
    srcs = ["model_server_config.proto"],
    cc_api_version = 2,
    deps = [
        ":file_system_storage_path_source_proto",
        ":logging_config_proto",
        "@com_google_protobuf//:cc_wkt_protos",
    ],
)

serving_proto_library_py(
    name = "model_server_config_proto_py_pb2",
    srcs = ["model_server_config.proto"],
    proto_library = "model_server_config_proto",
    deps = [
        "file_system_storage_path_source_proto_py_pb2",
        ":logging_config_proto_py_pb2",
    ],
)

serving_proto_library(
    name = "platform_config_proto",
    srcs = ["platform_config.proto"],
    cc_api_version = 2,
    deps = [
        "@com_google_protobuf//:cc_wkt_protos",
    ],
)

serving_proto_library_py(
    name = "platform_config_py_pb2",
    proto_library = "platform_config_proto",
)

serving_proto_library(
    name = "log_collector_config_proto",
    srcs = ["log_collector_config.proto"],
    cc_api_version = 2,
    deps = [
    ],
)

serving_proto_library_py(
    name = "log_collector_config_proto_py_pb2",
    srcs = ["log_collector_config.proto"],
    proto_library = "log_collector_config_proto",
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

serving_proto_library_py(
    name = "logging_config_proto_py_pb2",
    srcs = ["logging_config.proto"],
    proto_library = "logging_config_proto",
    deps = [
        ":log_collector_config_proto_py_pb2",
    ],
)

serving_proto_library(
    name = "monitoring_config_proto",
    srcs = ["monitoring_config.proto"],
    cc_api_version = 2,
    deps = [
    ],
)

serving_proto_library(
    name = "ssl_config_proto",
    srcs = ["ssl_config.proto"],
    cc_api_version = 2,
    deps = [
    ],
)
