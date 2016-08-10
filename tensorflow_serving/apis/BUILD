# Description: Tensorflow Serving APIs.

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
load("//tensorflow_serving:serving.bzl", "serving_proto_library_py")

serving_proto_library(
    name = "model_proto",
    srcs = ["model.proto"],
    cc_api_version = 2,
    go_api_version = 2,
    java_api_version = 2,
    deps = [
        "@protobuf//:cc_wkt_protos",
    ],
)

serving_proto_library_py(
    name = "model_proto_py_pb2",
    srcs = ["model.proto"],
    proto_library = "model_proto",
    deps = [],
)

serving_proto_library(
    name = "predict_proto",
    srcs = ["predict.proto"],
    cc_api_version = 2,
    go_api_version = 2,
    java_api_version = 2,
    deps = [
        ":model_proto",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

serving_proto_library_py(
    name = "predict_proto_py_pb2",
    srcs = ["predict.proto"],
    proto_library = "predict_proto",
    deps = [
        ":model_proto_py_pb2",
        "@org_tensorflow//tensorflow/core:protos_all_py",
    ],
)

serving_proto_library(
    name = "prediction_service_proto",
    srcs = ["prediction_service.proto"],
    has_services = 1,
    cc_api_version = 2,
    cc_grpc_version = 1,
    go_api_version = 2,
    java_api_version = 2,
    deps = [
        ":predict_proto",
    ],
)

py_library(
    name = "prediction_service_proto_py_pb2",
    srcs = ["prediction_service_pb2.py"],
    deps = [":predict_proto_py_pb2"],
)
