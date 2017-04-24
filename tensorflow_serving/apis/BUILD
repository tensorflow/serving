# Description: Tensorflow Serving APIs.

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
load("//tensorflow_serving:serving.bzl", "serving_proto_library_py")
load("//tensorflow_serving:serving.bzl", "serving_go_grpc_library")

serving_proto_library(
    name = "get_model_metadata_proto",
    srcs = ["get_model_metadata.proto"],
    cc_api_version = 2,
    go_api_version = 2,
    java_api_version = 2,
    deps = [
        ":model_proto",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@protobuf//:cc_wkt_protos",
    ],
)

serving_proto_library_py(
    name = "get_model_metadata_proto_py_pb2",
    srcs = ["get_model_metadata.proto"],
    proto_library = "get_model_metadata_proto",
    deps = [
        ":model_proto_py_pb2",
        "@org_tensorflow//tensorflow/core:protos_all_py",
    ],
)

serving_proto_library(
    name = "input_proto",
    srcs = ["input.proto"],
    cc_api_version = 2,
    go_api_version = 2,
    java_api_version = 2,
    deps = [
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@protobuf//:cc_wkt_protos",
    ],
)

serving_proto_library_py(
    name = "input_proto_py_pb2",
    srcs = ["input.proto"],
    proto_library = "input_proto",
    deps = [
        "@org_tensorflow//tensorflow/core:protos_all_py",
    ],
)

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
        ":classification_proto",
        ":get_model_metadata_proto",
        ":inference_proto",
        ":predict_proto",
        ":regression_proto",
    ],
)

py_library(
    name = "prediction_service_proto_py_pb2",
    srcs = ["prediction_service_pb2.py"],
    deps = [
        ":classification_proto_py_pb2",
        ":get_model_metadata_proto_py_pb2",
        ":inference_proto_py_pb2",
        ":predict_proto_py_pb2",
        ":regression_proto_py_pb2",
    ],
)

serving_go_grpc_library(
    name = "prediction_service_grpc",
    srcs = [":prediction_service_proto"],
    deps = [":prediction_service_proto"],
)

serving_proto_library(
    name = "classification_proto",
    srcs = ["classification.proto"],
    cc_api_version = 2,
    go_api_version = 2,
    java_api_version = 2,
    deps = [
        ":input_proto",
        ":model_proto",
    ],
)

serving_proto_library_py(
    name = "classification_proto_py_pb2",
    srcs = ["classification.proto"],
    proto_library = "classification_proto",
    deps = [
        ":input_proto_py_pb2",
        ":model_proto_py_pb2",
        "@org_tensorflow//tensorflow/core:protos_all_py",
    ],
)

serving_proto_library(
    name = "inference_proto",
    srcs = ["inference.proto"],
    cc_api_version = 2,
    go_api_version = 2,
    java_api_version = 2,
    deps = [
        ":classification_proto",
        ":input_proto",
        ":model_proto",
        ":regression_proto",
    ],
)

serving_proto_library_py(
    name = "inference_proto_py_pb2",
    srcs = ["inference.proto"],
    proto_library = "inference_proto",
    deps = [
        ":classification_proto_py_pb2",
        ":input_proto_py_pb2",
        ":model_proto_py_pb2",
        ":regression_proto_py_pb2",
    ],
)

serving_proto_library(
    name = "regression_proto",
    srcs = ["regression.proto"],
    cc_api_version = 2,
    go_api_version = 2,
    java_api_version = 2,
    deps = [
        ":input_proto",
        ":model_proto",
    ],
)

serving_proto_library_py(
    name = "regression_proto_py_pb2",
    srcs = ["regression.proto"],
    proto_library = "regression_proto",
    deps = [
        ":input_proto_py_pb2",
        ":model_proto_py_pb2",
        "@org_tensorflow//tensorflow/core:protos_all_py",
    ],
)

cc_library(
    name = "classifier",
    hdrs = ["classifier.h"],
    deps = [
        ":classification_proto",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "regressor",
    hdrs = ["regressor.h"],
    deps = [
        ":regression_proto",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)
