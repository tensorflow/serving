# Description: Tensorflow Serving configs.

package(
    default_visibility = ["//visibility:public"],
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

load("//tensorflow_serving:serving.bzl", "serving_proto_library")

serving_proto_library(
    name = "model_server_config_proto",
    srcs = ["model_server_config.proto"],
    cc_api_version = 2,
    deps = [
        "@protobuf//:cc_wkt_protos",
    ],
)
